#include <cuda_runtime.h>
#include "../../cuda_common.h"
#include "./deformable.cuh"
#include "../../SPHHelper.h"

namespace msph {

extern __device__ MultiphaseParam   d_param;
extern __device__ SimDataMultiphase d_data;
extern __device__ SPHHelper         d_sphhelper;

template <typename F>
__device__ inline void forEachNeighbor(int i, cfloat3 pos, F lambda)
{

    cint3 gridPos = calcGridPos(pos, d_param.gridxmin, d_param.dx);

    for (int z = -1; z <= 1; z++)
        for (int y = -1; y <= 1; y++)
            for (int x = -1; x <= 1; x++)
            {

                cint3 nPos     = gridPos + cint3(x, y, z);             //neighbor cell index
                uint  gridHash = calcGridHash(nPos, d_param.gridres);  //neighbor cell hash
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

__global__ void computeInnerForce(int num_particles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= num_particles)
        return;
    if (d_data.type[index] != TYPE_DEFORMABLE)
        return;

    cfloat3 pos     = d_data.pos[index];
    float   sr      = d_param.smoothradius;
    float   sr2     = d_param.smoothradius * d_param.smoothradius;
    float   density = 0;
    cfloat3 force(0, 0, 0);
    forEachNeighbor(index, pos, [&](int j) {
        cfloat3 xj  = d_data.pos[j];
        cfloat3 xij = pos - xj;
        float   d   = xij.length();
        if (j == index || d_data.type[j] != TYPE_DEFORMABLE)
            return;
        if (d > sr)
            return;

        cfloat3 nablaw = d_sphhelper.CubicGradient(xij);
        //mvprod(d_data.L[index], nablaw, nablaw);
        cmat3 stress_ij = d_data.stress[index] * (d_data.mass[index] / d_data.density[index])
                          + d_data.stress[j] * (d_data.mass[j] / d_data.density[j]);
        cfloat3 f;
        mvprod(stress_ij, nablaw, f);
        force += f;
    });
    d_data.force[index] += force / d_data.density[index];
    //printf("%f %f %f\n", force.x, force.y, force.z);
}

void computeInnerForce_host(int num_particles)
{
    uint num_threads, num_blocks;
    computeBlockSize(num_particles, 256, num_blocks, num_threads);

    computeInnerForce<<<num_blocks, num_threads>>>(num_particles);
    cudaThreadSynchronize();
    getLastCudaError("Kernel failed: compute inner force");
}

__device__ inline void vonmises(cmat3& stress)
{
    float sqrtj2 = stress.length();
    sqrtj2 /= 1.4142135f;
    float yield_ = d_param.Yield;
    if (sqrtj2 > yield_)
    {
        stress *= yield_ / sqrtj2;
    }
}

__device__ inline void druckerprager(cmat3& stress, cmat3& D, float pressure, cmat3& d_stress, int index)
{

    cmat3 s  = stress;  //copy
    float I1 = (s[0][0] + s[1][1] + s[2][2]) / 3.0f;
    /*s[0][0] -= I1;
		s[1][1] -= I1;
		s[2][2] -= I1;*/
    float sqrtj2  = s.length() / 1.4142135f;
    float total_p = pressure;
    float surface = d_param.granularFriction * pressure;
    //if (sqrtj2 > surface && sqrtj2 > EPSILON) {
    //	float tmp = 0;
    //	for (int i = 0; i < 9; i++)
    //		tmp += s.data[i] * D.data[i];
    //
    //	//lambda
    //	float diag = D[0][0] + D[1][1] + D[2][2];
    //	float lambda = 3.0f * d_param.alpha_phi * d_param.solidK * diag / d_param.solidG +
    //		1.0f / sqrtj2 * tmp;
    //	for (int i = 0; i < 9; i++) {
    //		d_stress.data[i] -= s.data[i] * d_param.solidG / sqrtj2 * lambda * d_param.dt;
    //	}

    //}

    //scale back
    //s = stress; //copy
    if (sqrtj2 > surface && sqrtj2 > EPSILON)
    {
        for (int i = 0; i < 9; i++)
        {
            s.data[i] *= surface / sqrtj2;
        }
        stress = s;
        //d_data.color[index].x = 1;
    }
    else
    {
        //d_data.color[index].x = 0;
    }
}

__global__ void updateParticleState(int num_particles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= num_particles)
        return;
    if (d_data.type[index] != TYPE_DEFORMABLE)
        return;

    cfloat3 pos = d_data.pos[index];
    float   sr  = d_param.smoothradius;
    cmat3   vgrad;

    forEachNeighbor(index, pos, [&](int j) {
        cfloat3 xj  = d_data.pos[j];
        cfloat3 xij = pos - xj;
        float   d   = xij.length();
        if (j == index || d_data.type[j] != TYPE_DEFORMABLE)
            return;
        if (d > sr)
            return;
        cfloat3 vji    = d_data.vel[j] - d_data.vel[index];
        cfloat3 nablaw = d_sphhelper.CubicGradient(xij);
        //mvprod(d_data.L[index], nablaw, nablaw);
        cmat3 nablav = TensorProduct(vji, nablaw);
        nablav.Multiply(d_data.mass[j] / d_data.density[j]);
        mat3add(vgrad, nablav, vgrad);
    });

    //strain rate D
    cmat3 vgT, D, tau;
    mat3transpose(vgrad, vgT);
    D = (vgrad + vgT) * 0.5f;

    //cauchy stress
    //rotation part
    cmat3 Omega, OmegaT, tmp1, tmp2, d_stress;
    Omega = (vgrad - vgT) * 0.5f;
    mat3transpose(Omega, OmegaT);
    mat3prod(d_data.stress[index], OmegaT, tmp1);
    mat3prod(Omega, d_data.stress[index], tmp2);
    mat3add(tmp1, tmp2, tmp1);

    //shear part
    auto trace = (D[0][0] + D[1][1] + D[2][2]) / 3.0f;
    tmp2       = D;
    //tmp2[0][0] -= trace;
    //tmp2[1][1] -= trace;
    //tmp2[2][2] -= trace;
    tmp2.Multiply(d_param.solidG * 2.0f);
    //bulk part
    //float pressure = trace * 3.0f * d_param.solidK; //minus pressure
    //tmp2[0][0] += pressure;
    //tmp2[1][1] += pressure;
    //tmp2[2][2] += pressure;

    mat3add(d_stress, tmp1, d_stress);
    mat3add(d_stress, tmp2, d_stress);
    d_stress.Multiply(d_param.dt);

    mat3add(d_data.stress[index], d_stress, d_data.stress[index]);

    //drucker prager model
    //druckerprager(d_data.stress[index], D, d_data.pressure[index], d_stress,index);
    //von mises model
    //vonmises(d_data.stress[index]);
}

void updateParticleState_host(int num_particles)
{
    uint num_threads, num_blocks;
    computeBlockSize(num_particles, 256, num_blocks, num_threads);

    updateParticleState<<<num_blocks, num_threads>>>(num_particles);
    cudaThreadSynchronize();
    getLastCudaError("Kernel failed: update particle state");
}

__global__ void correctKernel(int num_particles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= num_particles)
        return;
    if (d_data.type[index] != TYPE_DEFORMABLE)
        return;

    cfloat3 pos = d_data.pos[index];
    float   sr  = d_param.smoothradius;
    cmat3   L;

    forEachNeighbor(index, pos, [&](int j) {
        cfloat3 xj  = d_data.pos[j];
        cfloat3 xij = pos - xj;
        float   d   = xij.length();
        if (j == index || d_data.type[j] != TYPE_DEFORMABLE)
            return;
        if (d > sr)
            return;

        cfloat3 nablaw = d_sphhelper.CubicGradient(xij);
        nablaw         = nablaw * d_data.mass[j] / d_data.density[j];
        cmat3 tmp      = TensorProduct(xj - pos, nablaw);
        mat3add(L, tmp, L);
    });
    float Ldet = L.Det();
    if (fabsf(Ldet) < 0.1f)
    {
        make_identity(L);
    }
    else
    {
        L = L.Inv();
    }
    //make_identity(L);
    d_data.L[index] = L;
}

void correctKernel_host(int num_particles)
{
    uint num_threads, num_blocks;
    computeBlockSize(num_particles, 256, num_blocks, num_threads);

    correctKernel<<<num_blocks, num_threads>>>(num_particles);
    cudaThreadSynchronize();
    getLastCudaError("Kernel failed: correct kernel");
}
};  // namespace msph