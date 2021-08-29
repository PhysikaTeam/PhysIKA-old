#include "multiphase_WCSPH.cuh"
#include "../../cuda_common.h"
#include "../../SPHHelper.h"
#include <device_launch_parameters.h>
#include "../../../math/geometry_helper.h"

namespace msph {

__device__ MultiphaseParam   d_param;
__device__ SimDataMultiphase d_data;
__device__ SPHHelper         d_sphhelper;
MultiphaseParam*             pParamStatic;

void CopyParamToDevice(MultiphaseParam& h_param)
{
    cudaMemcpyToSymbol(d_param, &h_param, sizeof(MultiphaseParam));

    SPHHelper sphHelper;
    sphHelper.SetupCubic(h_param.smoothradius * 0.5f);
    cudaMemcpyToSymbol(d_sphhelper, &sphHelper, sizeof(SPHHelper));
}

void CopyParamFromDevice(MultiphaseParam& h_param)
{
    cudaMemcpyFromSymbol(&h_param, d_param, sizeof(MultiphaseParam));
}

void copyDataPtrToDevice(const SimDataMultiphase& data)
{
    cudaMemcpyToSymbol(d_data, &data, sizeof(SimDataMultiphase));
}

template <typename F>
__device__ inline void forEachNeighbor(cfloat3 pos, MultiphaseParam* param, F lambda)
{

    cint3 gridPos = calcGridPos(pos, param->gridxmin, param->dx);

    for (int z = -1; z <= 1; z++)
        for (int y = -1; y <= 1; y++)
            for (int x = -1; x <= 1; x++)
            {

                cint3 nPos     = gridPos + cint3(x, y, z);            //neighbor cell index
                uint  gridHash = calcGridHash(nPos, param->gridres);  //neighbor cell hash
                if (gridHash == GRID_UNDEF)
                    continue;
                uint startIndex = d_data.gridCellStart[gridHash];
                if (startIndex == 0xffffffff)
                    continue;
                uint endIndex = d_data.gridCellEnd[gridHash];

                for (uint j = startIndex; j < endIndex; j++)
                {
                    lambda(j);
                }
            }
}

template <typename F>
__device__ inline void forEachNeighborInList(int index, F lambda)
{
    uint neighborCount    = d_data.neighborList[index * NUM_NEIGHBOR];
    uint neighborIdxStart = index * NUM_NEIGHBOR + 1;
    uint neighborIdxEnd   = neighborIdxStart + neighborCount;
    for (uint j = neighborIdxStart; j < neighborIdxEnd; j++)
    {
        lambda(d_data.neighborList[j]);
    }
}

HDFUNC __inline__ void TensorProductInplace(cfloat3& a, cfloat3& b, cmat3& res)
{
    res[0][0] = a.x * b.x;
    res[0][1] = a.x * b.y;
    res[0][2] = a.x * b.z;
    res[1][0] = a.y * b.x;
    res[1][1] = a.y * b.y;
    res[1][2] = a.y * b.z;
    res[2][0] = a.z * b.x;
    res[2][1] = a.z * b.y;
    res[2][2] = a.z * b.z;
}

__device__ inline void ApplyDruckerPrager(cmat3& shearStress, float p, MultiphaseParam* param)
{
    auto& s   = shearStress;
    float tau = s.length() / 1.4142135f;

    float tmpP    = p;
    float surface = param->granularFriction * tmpP + param->cohesion;
    if (surface < 0)
        surface = 0;
    if (tau > surface && tau > 1e-10)
    {
        for (int i = 0; i < 9; i++)
        {
            s.data[i] *= surface / tau;
        }
    }
}

__global__ void UpdateParticleState(int num_particles, MultiphaseParam* param)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_particles)
        return;

    index = d_data.particleIndex[index];
    if (d_data.type[index] == TYPE_RIGID)
        return;

    cfloat3 pos = d_data.pos[index];

    float   sr      = param->smoothradius;
    float   sr2     = sr * sr;
    double  density = 0;
    cmat3   vGradient;
    cmat3   nablav;
    float   vol0  = param->spacing * param->spacing * param->spacing;
    auto&   volfi = d_data.vFrac[index];
    volf    update;
    cfloat3 vi = d_data.vel[index];

    float diffusionFac = 2.0f * param->drift_dynamic_diffusion * vol0;

    for (int k = 0; k < param->numTypes; k++)
    {
        d_data.vfrac_change[index].data[k] = 0;
        update.data[k]                     = 0;
    }
    forEachNeighbor(pos, param, [&](int j) {
        j           = d_data.particleIndex[j];
        cfloat3 xj  = d_data.pos[j];
        cfloat3 xij = pos - xj;
        float   d   = xij.length();
        if (d > sr)
            return;
        float   wij    = d_sphhelper.Cubic(d);
        cfloat3 nablaw = d_sphhelper.CubicGradient(xij);
        auto&   volfj  = d_data.vFrac[j];

        if (d_data.type[j] != TYPE_RIGID)
        {
            density += wij;
            cfloat3 vji = (d_data.vel[j] - vi) * vol0;
            auto&   a   = vji;
            auto&   b   = nablaw;

            //if (d_data.type[index] == d_data.type[j])
            //{
            auto& res = vGradient;
            res[0][0] += a.x * b.x;
            res[0][1] += a.x * b.y;
            res[0][2] += a.x * b.z;
            res[1][0] += a.y * b.x;
            res[1][1] += a.y * b.y;
            res[1][2] += a.y * b.z;
            res[2][0] += a.z * b.x;
            res[2][1] += a.z * b.y;
            res[2][2] += a.z * b.z;
            //}

            if (param->dissolution)
            {
                float diffFactor = diffusionFac * dot(xij, nablaw) / (d * d + 0.01f);
                for (int k = 0; k < param->numTypes; k++)
                {
                    update.data[k] += diffFactor * (volfi.data[k] - volfj.data[k]);
                }
            }
        }
        else if (d_data.type[j] == TYPE_RIGID)
        {
            density += d_data.restDensity[j] * wij;
        }
    });

    if (param->dissolution)
    {
        for (int k = 0; k < param->numTypes; k++)
        {
            d_data.vfrac_change[index].data[k] = update.data[k];
        }
    }
    density *= d_data.mass[index];
    d_data.density[index]  = density;
    d_data.pressure[index] = param->pressureK * (powf(density / d_data.restDensity[index], 7.0f) - 1.0f) * d_data.restDensity[index];
    if (d_data.pressure[index] < 0)
        d_data.pressure[index] = 0;

    cmat3 D, Omega;
    cmat3 tmp1, tmp2;
    cmat3 sigma = d_data.stress[index];
    mat3transpose(vGradient, tmp1);

    for (int k = 0; k < 9; k++)
    {
        D.data[k]     = (vGradient.data[k] + tmp1.data[k]) * 0.5f;
        Omega.data[k] = (vGradient.data[k] - tmp1.data[k]) * 0.5f;
    }
    mat3prod(sigma, Omega, tmp1);
    mat3prod(Omega, sigma, tmp2);
    for (int k = 0; k < 9; k++)
        tmp1.data[k] = -tmp1.data[k] + tmp2.data[k];
    //shear part
    auto trace = (D[0][0] + D[1][1] + D[2][2]) / 3.0f;
    D[0][0] -= trace;
    D[1][1] -= trace;
    D[2][2] -= trace;
    for (int k = 0; k < 9; k++)
        tmp2.data[k] = D.data[k] * (param->solidG * 2.0f);
    //bulk part
    trace *= 3.0f * d_param.solidK;  //minus pressure
    tmp2[0][0] += trace;
    tmp2[1][1] += trace;
    tmp2[2][2] += trace;

    for (int k = 0; k < 9; k++)
        sigma.data[k] += (tmp1.data[k] + tmp2.data[k]) * param->dt;

    cmat3 shearStress = sigma;
    float pressure    = -(shearStress[0][0] + shearStress[1][1] + shearStress[2][2]) / 3.0f;
    shearStress[0][0] += pressure;
    shearStress[1][1] += pressure;
    shearStress[2][2] += pressure;
    ApplyDruckerPrager(shearStress, pressure, param);
    sigma = shearStress;
    if (pressure < 0)
        pressure = 0;  // tensile crack
    sigma[0][0] -= pressure;
    sigma[1][1] -= pressure;
    sigma[2][2] -= pressure;
    d_data.stress[index] = sigma;

    for (int k = 0; k < 9; k++)
    {
        d_data.stressMix[index].data[k] = sigma.data[k] * d_data.vFrac[index].data[1]
                                          + D.data[k] * param->viscosity * d_data.vFrac[index].data[0];
    }
    d_data.stressMix[index][0][0] -= d_data.pressure[index] * d_data.vFrac[index].data[0];
    d_data.stressMix[index][1][1] -= d_data.pressure[index] * d_data.vFrac[index].data[0];
    d_data.stressMix[index][2][2] -= d_data.pressure[index] * d_data.vFrac[index].data[0];
}

void UpdateParticleStateHost(int numParticles)
{
    uint num_threads, num_blocks;
    computeBlockSize(numParticles, 256, num_blocks, num_threads);
    UpdateParticleState<<<num_blocks, num_threads>>>(numParticles, pParamStatic);
    cudaThreadSynchronize();
    getLastCudaError("Kernel failed: compute pressure multiphase");
}

HDFUNC __inline__ void MatVecProd(cmat3& m, cfloat3& v, cfloat3& c)
{
    auto& tmp = c;
    tmp.x     = m.data[0] * v.x + m.data[1] * v.y + m.data[2] * v.z;
    tmp.y     = m.data[3] * v.x + m.data[4] * v.y + m.data[5] * v.z;
    tmp.z     = m.data[6] * v.x + m.data[7] * v.y + m.data[8] * v.z;
}

__global__ void ComputeForceByTensor(int nump, MultiphaseParam* param)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nump)
        return;
    index = d_data.particleIndex[index];
    if (d_data.type[index] == TYPE_RIGID)
        return;

    cfloat3 pos = d_data.pos[index];
    cfloat3 force(0, 0, 0);
    float   sr  = param->smoothradius;
    float   sr2 = sr * sr;

    cfloat3 sften(0, 0, 0);
    cfloat3 viscf_b(0, 0, 0);
    float   vol0 = param->spacing * param->spacing * param->spacing;

    float mass_tmp = d_data.mass[index];
    float pi       = d_data.pressure[index];

    cfloat3 xsph(0, 0, 0);
    cmat3   sij;
    cfloat3 f;
    cmat3   si = d_data.stressMix[index];
    cfloat3 vij;
    cfloat3 nablawij;
    cfloat3 fVisc(0, 0, 0);

    forEachNeighbor(pos, param, [&](int j) {
        j           = d_data.particleIndex[j];
        cfloat3 xj  = d_data.pos[j];
        cfloat3 xij = pos - xj;
        float   d   = xij.length();
        if (d >= sr || j == index)
            return;

        vij      = d_data.vel[index] - d_data.vel[j];
        nablawij = d_sphhelper.CubicGradient(xij);

        if (d_data.type[j] != TYPE_RIGID)
        {
            for (int k = 0; k < 9; k++)
                sij.data[k] = si.data[k] + d_data.stressMix[j].data[k];
            MatVecProd(sij, nablawij, f);
            force += f;

            //artificial viscosity

            float xv = dot(vij, xij);
            if (xv < 0)
            {
                float phi = sr * xv / (d * d + 0.0025f * sr2);
                float factor;

                if (d_data.type[j] == d_data.type[index])
                    factor = param->solid_visc * phi * vol0;
                else
                    factor = param->dragForce * phi * vol0;
                fVisc += nablawij * factor;
            }
        }

        if (d_data.type[j] == TYPE_RIGID)
        {

            float pc = pi * d_data.restDensity[j];
            force += nablawij * pc * (-1.0f);
            float xv    = dot(vij, xij);
            float massj = d_data.restDensity[j] * d_data.mass[index];
            if (xv < 0)
            {
                fVisc += nablawij * (xv / (d * d + 0.0025f * sr2) * param->boundary_visc * sr / d_data.density[index] * massj);
            }
        }
    });
    force = force * (vol0 * vol0 / mass_tmp);
    force.y += -9.8f;
    force -= xsph * (param->viscosity / param->dt);
    force += fVisc;
    d_data.force[index] = force;
}

void computeForce_host(int num_p)
{
    uint num_threads, num_blocks;
    computeBlockSize(num_p, 256, num_blocks, num_threads);

    //ComputeForce << <num_blocks, num_threads >> > (num_p, pParamStatic);
    ComputeForceByTensor<<<num_blocks, num_threads>>>(num_p, pParamStatic);
    cudaThreadSynchronize();
    getLastCudaError("Kernel failed: compute force multiphase");
}

__global__ void advectParticles(int num_p, MultiphaseParam* param)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= num_p)
        return;
    if (d_data.type[index] == TYPE_RIGID && d_data.group[index] == GROUP_FIXED)
        return;
    d_data.vel[index] += d_data.force[index] * param->dt;
    d_data.pos[index] += d_data.vel[index] * param->dt;
    d_data.drift_accel[index] = param->gravity - d_data.force[index];

    if (param->dissolution)
    {
        for (int k = 0; k < param->numTypes; k++)
        {
            d_data.vFrac[index].data[k] += d_data.vfrac_change[index].data[k] * param->dt;
        }
        d_data.mass[index] = 0;
        for (int k = 0; k < param->numTypes; k++)
        {
            d_data.mass[index] += d_data.vFrac[index].data[k] * param->densArr[k] * param->vol0;
            d_data.color[index][k] = d_data.vFrac[index].data[k];
        }
    }
}

void advectParticles_host(int num_p)
{
    uint num_threads, num_blocks;
    computeBlockSize(num_p, 256, num_blocks, num_threads);

    advectParticles<<<num_blocks, num_threads>>>(num_p, pParamStatic);
    cudaThreadSynchronize();
    getLastCudaError("Kernel failed: advect particles");
}

//diffusion
__global__ void computeDriftVel(int num_particles, MultiphaseParam* param)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= num_particles)
        return;
    if (d_data.type[index] != TYPE_FLUID)
        return;

    float    rest_density = d_data.restDensity[index];
    auto     C_m          = param->drift_dynamic_diffusion;
    auto&    vol_frac     = d_data.vFrac[index];
    cfloat3* drift_v      = &d_data.drift_v[index * param->numTypes];
    cfloat3  drift_accel  = d_data.drift_accel[index];

    for (int k = 0; k < param->numTypes; k++)
    {  //for each phase
        if (vol_frac.data[k] < EPSILON)
        {
            drift_v[k] = cfloat3(0, 0, 0);
            continue;
        }
        float   density_k      = param->densArr[k];
        float   density_factor = (density_k - rest_density) / rest_density;
        cfloat3 drift_vk       = drift_accel * C_m * density_factor;
        drift_v[k]             = drift_vk;
    }
}

void computeDriftVel_host(int num_particles)
{
    uint num_threads, num_blocks;
    computeBlockSize(num_particles, 256, num_blocks, num_threads);

    computeDriftVel<<<num_blocks, num_threads>>>(num_particles, pParamStatic);
    cudaThreadSynchronize();
    getLastCudaError("Kernel failed: compute force multiphase");
}

__global__ void compupteInterPhaseTensor(int num_particles, MultiphaseParam* param)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= num_particles)
        return;
    if (d_data.type[index] != TYPE_FLUID)
        return;

    auto&    vol_frac     = d_data.vFrac[index];
    auto&    drift_accel  = d_data.drift_accel[index];
    auto&    rest_density = d_data.restDensity[index];
    cfloat3* drift_v      = &d_data.drift_v[index * param->numTypes];
    auto&    M_m          = d_data.M_m[index];
    auto&    umkumk       = d_data.umkumk[index];
    M_m                   = cfloat3(0, 0, 0);
    umkumk.Set(0.0f);

    for (int k = 0; k < param->numTypes; k++)
    {  //for each phase
        if (vol_frac.data[k] < EPSILON)
            continue;
        M_m += drift_accel * vol_frac.data[k] * (param->densArr[k] - rest_density) / param->densArr[k];
        umkumk += TensorProduct(drift_v[k], drift_v[k]) * (vol_frac.data[k] * (-1.0f));
    }
}
void computeInterPhaseTensor_host(int num_particles)
{
    uint num_threads, num_blocks;
    computeBlockSize(num_particles, 256, num_blocks, num_threads);

    compupteInterPhaseTensor<<<num_blocks, num_threads>>>(num_particles, pParamStatic);
    cudaThreadSynchronize();
    getLastCudaError("Kernel failed: compute force multiphase");
}

__global__ void computePhaseDiffusion(int num_particles, MultiphaseParam* param)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= num_particles)
        return;
    if (d_data.type[index] != TYPE_FLUID)
        return;

    float    sr           = param->smoothradius;
    float    sr2          = sr * sr;
    int      num_type     = param->numTypes;
    cfloat3  xi           = d_data.pos[index];
    cfloat3* drift_v      = &d_data.drift_v[index * param->numTypes];
    auto&    vfrac_change = d_data.vfrac_change[index];
    for (int k = 0; k < param->numTypes; k++)
    {
        vfrac_change.data[k] = 0.0f;
    }
    float D_m  = param->drift_turbulent_diffusion;
    float vol0 = param->spacing * param->spacing * param->spacing;
    forEachNeighbor(xi, param, [&](int j) {
        if (d_data.type[j] != TYPE_FLUID || j == index)
            return;

        cfloat3 xj  = d_data.pos[j];
        cfloat3 xij = xi - xj;
        float   d   = xij.length();
        if (d >= sr)
            return;

        cfloat3 nabla_w = d_sphhelper.CubicGradient(xij);
        float   vol_j   = d_data.mass[j] / d_data.density[j];
        float   fac     = dot(xij, nabla_w) / (d * d + 0.01f * sr2 * 0.25f) * 2.0f;

        bool enable_separation = true;
        for (int k = 0; k < num_type; k++)
        {
            //test separation
            cfloat3 flux_k = d_data.drift_v[index * num_type + k] * d_data.vFrac[index].data[k]
                             + d_data.drift_v[j * num_type + k] * d_data.vFrac[j].data[k];
            float tmpk = dot(flux_k, nabla_w) * vol0 * param->dt * (-1);
            if (d_data.vFrac[index].data[k] + tmpk < 0 || d_data.vFrac[j].data[k] - tmpk < 0)
            {
                enable_separation = false;
            }
        }
        for (int k = 0; k < num_type; k++)
        {
            if (enable_separation)
            {
                cfloat3 flux_k = d_data.drift_v[index * num_type + k] * d_data.vFrac[index].data[k]
                                 + d_data.drift_v[j * num_type + k] * d_data.vFrac[j].data[k];
                vfrac_change.data[k] -= dot(flux_k, nabla_w) * vol0 * param->dt;
            }
            //diffusion
            float factmp = fac * (d_data.vFrac[index].data[k] - d_data.vFrac[j].data[k]);
            vfrac_change.data[k] += factmp * D_m * vol0 * param->dt;
        }
    });
}
HDFUNC inline void clamp(float& x, float min, float max)
{
    if (x < min)
        x = min;
    if (x > max)
        x = max;
    return;
}

__global__ void updateVolumeFraction(int num_particles, MultiphaseParam* param)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= num_particles)
        return;
    if (d_data.type[index] != TYPE_FLUID)
        return;

    int   num_type     = param->numTypes;
    auto& vfrac        = d_data.vFrac[index];
    auto& vfrac_change = d_data.vfrac_change[index];
    float sum          = 0;
    float sumerr       = 0;
    for (int k = 0; k < num_type; k++)
    {
        vfrac.data[k] += vfrac_change.data[k];
        sumerr += vfrac.data[k];
        clamp(vfrac.data[k], 0, 1);
        sum += vfrac.data[k];
    }
    for (int k = 0; k < num_type; k++)
    {
        vfrac.data[k] /= sum;
    }
}

void computePhaseDiffusion_host(SimDataMultiphase& data, int num_particles, int num_types)
{
    uint num_threads, num_blocks;
    computeBlockSize(num_particles, 256, num_blocks, num_threads);

    computePhaseDiffusion<<<num_blocks, num_threads>>>(num_particles, pParamStatic);
    cudaThreadSynchronize();
    getLastCudaError("Kernel failed: compute phase diffusion");

    updateVolumeFraction<<<num_blocks, num_threads>>>(num_particles, pParamStatic);
    cudaThreadSynchronize();
    getLastCudaError("Kernel failed: compute phase diffusion");

    float* dbg_pt = new float[num_particles * num_types];
    cudaMemcpy(dbg_pt, data.vFrac, num_particles * num_types * sizeof(float), cudaMemcpyDeviceToHost);
    float verify = 0;
    for (int i = 0; i < num_particles; i++)
        verify += dbg_pt[i * num_types];
    printf("total volume fraction phase 0: %f\n", verify);
    delete dbg_pt;
}

__global__ void updateMass(int num_particles, MultiphaseParam* param)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= num_particles)
        return;
    if (d_data.type[index] == TYPE_RIGID)
        return;
    float visc         = 0;
    float rest_density = 0;
    auto& vol_frac     = d_data.vFrac[index];

    for (int k = 0; k < param->numTypes; k++)
    {
        rest_density += vol_frac.data[k] * param->densArr[k];
        visc += vol_frac.data[k] * param->viscArr[k];
    }

    d_data.mass[index]        = rest_density * param->vol0;
    d_data.viscosity[index]   = visc;
    d_data.restDensity[index] = rest_density;
    d_data.density[index]     = rest_density;
    d_data.color[index]       = cfloat4(vol_frac.data[0], vol_frac.data[1], 0, 1.0f);
}

void updateMass_host(int num_particles)
{
    uint num_threads, num_blocks;
    computeBlockSize(num_particles, 256, num_blocks, num_threads);
    updateMass<<<num_blocks, num_threads>>>(num_particles, pParamStatic);
    cudaThreadSynchronize();
    getLastCudaError(" failed: update mass");
}

__global__ void RigidParticleVolume(int num_particles, MultiphaseParam* param)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= num_particles)
        return;
    index = d_data.particleIndex[index];
    if (d_data.type[index] != TYPE_RIGID)
        return;

    cfloat3 xi         = d_data.pos[index];
    float   sr         = param->smoothradius;
    cint3   cell_index = calcGridPos(xi, param->gridxmin, param->dx);
    float   sumW       = 0;

    forEachNeighbor(xi, param, [&](int j) {
        j = d_data.particleIndex[j];
        if (d_data.type[j] != TYPE_RIGID)
            return;

        cfloat3 xj  = d_data.pos[j];
        cfloat3 xij = xi - xj;
        auto    d   = xij.length();
        if (d >= sr)
            return;

        sumW += d_sphhelper.Cubic(d);
        ;
    });
    d_data.restDensity[index] = 1 / sumW / param->vol0;  //effective volume
}

void computeRigidVolume_host(int num_particles)
{
    uint num_threads, num_blocks;
    computeBlockSize(num_particles, 256, num_blocks, num_threads);

    RigidParticleVolume<<<num_blocks, num_threads>>>(num_particles, pParamStatic);
    cudaThreadSynchronize();
    getLastCudaError(" execution failed: rigid particle volume");
}

__global__ void BuildNeighborList(int numParticles, MultiphaseParam* param)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
        return;
    index = d_data.particleIndex[index];

    cfloat3 xi            = d_data.pos[index];
    float   sr            = param->smoothradius;
    cint3   cell_index    = calcGridPos(xi, param->gridxmin, param->dx);
    int     neighborCount = 0;
    int     neighborIdx   = index * NUM_NEIGHBOR + 1;
    forEachNeighbor(xi, param, [&](int j) {
        j      = d_data.particleIndex[j];
        auto d = (xi - d_data.pos[j]).length();
        if (d >= sr)
            return;
        if (neighborCount >= NUM_NEIGHBOR - 2)
        {
            printf("error neighbor overflow\n");
            return;
        }
        d_data.neighborList[neighborIdx + neighborCount] = j;
        neighborCount++;
    });
    d_data.neighborList[index * NUM_NEIGHBOR] = neighborCount;
}

void BuildNeighborListHost(int numParticles)
{
    uint numThreads, numBlocks;
    computeBlockSize(numParticles, 256, numBlocks, numThreads);

    BuildNeighborList<<<numBlocks, numThreads>>>(numParticles, pParamStatic);
    cudaThreadSynchronize();
    getLastCudaError(" execution failed: rigid particle volume");
}
};  // namespace msph
