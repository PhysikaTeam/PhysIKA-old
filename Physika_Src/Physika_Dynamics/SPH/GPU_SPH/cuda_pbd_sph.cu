/*
 * @file cuda_pbd_sph.cu
 * @Brief class CudaPBDSPH
 * @author Wei Chen
 *
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0.
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */


#include <sstream>
#include <iostream>
#include <fstream>
#include <string>

#include "cuda_runtime.h"
#include "device_atomic_functions.h"

#include "Physika_Core/Timer/timer.h"
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Core/Utilities/cuda_math_utilities.h"
#include "Physika_Core/Utilities/cuda_helper_math.h"

#include "Physika_Dynamics/SPH/GPU_SPH/cuda_pbd_sph.h"
#include "Physika_Dynamics/SPH/GPU_SPH/kernels.h"

namespace Physika {

#define   EPSILON   1e-6

__constant__ Settings PARAMS;
__constant__ SpikyKernel kernSpiky;
__constant__ CubicKernel kernCubic;
__constant__ SmoothKernel kernSmooth;

bool CudaPBDSPH::initialize(std::string in_filename)
{
    allocateMemory();

    grid_hash_.setSpace(2 * params_.samplingDistance, params_.lowBound, params_.upBound);

    std::vector<float3> poss;
    std::vector<float3> vels;
    for (int m = 0; m < models_.size(); m++)
    {
        int num = models_[m]->positions.size();
        for (int i = 0; i < num; i++)
        {
            poss.push_back(models_[m]->positions[i]);
            vels.push_back(models_[m]->velocities[i]);
        }
    }

    pos_arr_.copyFromHost(poss);
    vel_arr_.copyFromHost(vels);

    poss.clear();
    vels.clear();

    cudaCheck(cudaMemcpyToSymbol(PARAMS, &params_, sizeof(Settings)));

    computeNeighbors();
    computeDensity();

    float maxRho = 0.0f;

    float* bufArr = new float[params_.pNum];
    cudaCheck(cudaMemcpy(bufArr, rho_arr_.data(), params_.pNum * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < params_.pNum; i++)
    {
        if (bufArr[i] > maxRho)
            maxRho = bufArr[i];
    }

    params_.mass *= params_.restDensity / maxRho;
    std::cout << "Mass per particle: " << params_.mass << std::endl;

    initialSceneBoundary();

    return true;
}

void CudaPBDSPH::initialSceneBoundary()
{
    CudaDistanceField * box = new CudaDistanceField();
    box->setSpace(params_.lowBound - params_.samplingDistance * 5, params_.upBound + params_.samplingDistance * 5, 205, 105, 105);
    box->distanceFieldToBox(params_.lowBound, params_.upBound, true);

    //box->DistanceFieldToSphere(make_float3(0.5f, 0.5f, 0.5f), 0.5f, true);

    Barrier* b = new BarrierCudaDistanceField(box);
    b->setNormalFriction(params_.normalFriction);
    b->setTangentialFriction(params_.tangentialFriction);

    boundary_.insertBarrier(b);
}

void CudaPBDSPH::step(float dt)
{
    cudaCheck(cudaMemcpyToSymbol(PARAMS, &params_, sizeof(Settings)));

    //==================================================================================

    Timer timer;
    timer.startTimer();

    computeNeighbors();
    predict(dt);

    timer.stopTimer();
    std::cout << "time cost for predict: " << timer.getElapsedTime() << std::endl;

    //==================================================================================

    timer.startTimer();

    correctWithPBD(dt);

    timer.stopTimer();
    std::cout << "time cost for resolve incompressibility: " << timer.getElapsedTime() << std::endl;

    //==================================================================================

    timer.startTimer();

    applyViscosity(dt);
    boundaryHandling(dt);

    timer.stopTimer();
    std::cout << "time cost for resolve viscosity & boundary handling: " << timer.getElapsedTime() << std::endl;

    //cudaMemcpy(data, posArr.data, params.pNum * sizeof(float3), cudaMemcpyDeviceToHost);
}

__global__ void K_SetupRendering(float3* dst, float4* color, CudaArray<float3> posArr, CudaArray<float> colorIndex)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= posArr.size()) return;

    dst[pId] = posArr[pId];

    float a = 0.0f;
    float b = 0.0f;

    if (colorIndex[pId] > 0.0f)
        a = colorIndex[pId] / 100000.0f;

    if (a > 1.0f)
        a = 1.0f;

    if (colorIndex[pId] < 0.0f)
        b = -colorIndex[pId] / 100000.0f;

    if (b > 1.0f)
        b = 1.0f;

    color[pId] = colorIndex[pId] > 0.0f ? make_float4(1.0 - a, 1.0f, 0.0f, 1.0f) : make_float4(1.0f, 1.0 - b, 0.0f, 1.0f);

}

__global__ void K_Predict(CudaArray<float3> posArr, CudaArray<float3> velArr, float dt)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= posArr.size()) return;

    float3 pos_i = posArr[pId];
    float3 vel_i = velArr[pId];

    vel_i += make_float3(0.0f, -2.0f, 0.0f)*dt;

    //predict position x* = xi + dt * vi
    pos_i += vel_i*dt;

    posArr[pId] = pos_i;
    velArr[pId] = vel_i;
}

void CudaPBDSPH::predict(float dt)
{
    dim3 pDims = int(ceil(pos_arr_.size() / BLOCK_SIZE + 0.5f));
    K_Predict<<<pDims, BLOCK_SIZE >>> (pos_arr_, vel_arr_, dt);
}

__global__ void K_ConstrainPosition(CudaArray<float3> posArr, CudaArray<float3> velArr, float3 lo, float3 hi)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= posArr.size()) return;

    float3 pos = posArr[pId];
    float3 vel = velArr[pId];

    if (pos.x < lo.x) {
        vel.x = 0;
        pos.x = lo.x;
    }
    else if (pos.x > hi.x) {
        vel.x = 0;
        pos.x = hi.x;
    }

    if (pos.y < lo.y) {
        vel.y = 0;
        pos.y = lo.y;
    }
    else if (pos.y > hi.y) {
        vel.y = 0;
        pos.y = hi.y;
    }

    if (pos.z < lo.z) {
        vel.z = 0;
        pos.z = lo.z;
    }
    else if (pos.z > hi.z) {
        vel.z = 0;
        pos.z = hi.z;
    }

    posArr[pId] = pos;
    velArr[pId] = vel;
}


__global__ void K_ComputeLambdas(CudaArray<float> lambdaArr, CudaArray<float> rhoArr, CudaArray<float3> posArr, CudaArray<NeighborList> neighbors)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= posArr.size()) return;

    float3 pos_i = posArr[pId];

    float lamda_i = 0.0f;
    float3 grad_ci = make_float3(0.0f, 0.0f, 0.0f);

    int nbSize = neighbors[pId].size();
    for (int ne = 0; ne < nbSize; ne++)
    {
        int j = neighbors[pId][ne];
        float r = length(pos_i - posArr[j]);

        if (r > EPSILON)
        {
            float3 g = PARAMS.mass*kernSpiky.gradient(r, PARAMS.smoothingLength)*(pos_i - posArr[j]) * (1.0f / r);
            grad_ci += g;
            lamda_i += dot(g, g);
        }
    }

    lamda_i += dot(grad_ci, grad_ci);

    float rho_i = rhoArr[pId];

    lamda_i = -(rho_i - 1000.0f) / (lamda_i + 0.1f);

    lambdaArr[pId] = lamda_i > 0.0f ? 0.0f : lamda_i;
}

__global__ void K_ComputeDisplacement(CudaArray<float3> dPos, CudaArray<float> lambdas, CudaArray<float3> posArr, CudaArray<NeighborList> neighbors, float dt)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= posArr.size()) return;

    float3 pos_i = posArr[pId];

    float3 dpos_i = make_float3(0.0f, 0.0f, 0.0f);
    float lamda_i = lambdas[pId];

    int nbSize = neighbors[pId].size();
    for (int ne = 0; ne < nbSize; ne++)
    {
        int j = neighbors[pId][ne];
        float r = length(pos_i - posArr[j]);

        if (r > EPSILON)
        {
            //float3 dp_ij = 0.005f*0.5f*(lamda_i + lambdas[j])*PARAMS.mass*kernSpiky.Gradient(r, PARAMS.smoothingLength)*(pos_i - posArr[j]) * (1.0f / r);
            float3 dp_ij = 0.005f*0.5f*(lamda_i + lambdas[j])*PARAMS.mass*kernSpiky.gradient(r, PARAMS.smoothingLength)*(pos_i - posArr[j]) * (1.0f / r);
            dpos_i += dp_ij;
            atomicAdd(&dPos[pId].x, dp_ij.x);
            atomicAdd(&dPos[pId].y, dp_ij.y);
            atomicAdd(&dPos[pId].z, dp_ij.z);

            atomicAdd(&dPos[j].x, -dp_ij.x);
            atomicAdd(&dPos[j].y, -dp_ij.y);
            atomicAdd(&dPos[j].z, -dp_ij.z);
        }
    }
    //		dPos[pId] = dpos_i;
}


__global__ void K_UpdatePosition(CudaArray<float3> posArr, CudaArray<float3> velArr, CudaArray<float3> dPos, float dt)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= posArr.size()) return;

    posArr[pId] += dPos[pId];
    velArr[pId] += dPos[pId] / dt;
}

void CudaPBDSPH::iterationPBD(float dt)
{
    dim3 pDims = int(ceil(pos_arr_.size() / BLOCK_SIZE + 0.5f));

    K_ComputeLambdas<<<pDims, BLOCK_SIZE >>> (lambda_arr_, rho_arr_, pos_arr_, neighbors_arr_);
    buffer_.reset();
    K_ComputeDisplacement <<<pDims, BLOCK_SIZE >>> (buffer_, lambda_arr_, pos_arr_, neighbors_arr_, dt);
    K_UpdatePosition <<<pDims, BLOCK_SIZE >>> (pos_arr_, vel_arr_, buffer_, dt);
}

__global__ void K_ComputeDensity(CudaArray<float> rhoArr, CudaArray<float3> posArr, CudaArray<NeighborList> neighbors)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= posArr.size()) return;

    float r;
    float rho_i = 0.0f;
    float3 pos_i = posArr[pId];
    int nbSize = neighbors[pId].size();
    for (int ne = 0; ne < nbSize; ne++)
    {
        int j = neighbors[pId][ne];
        r = length(pos_i - posArr[j]);

        rho_i += PARAMS.mass*kernSpiky.weight(r, PARAMS.smoothingLength);
    }
    rhoArr[pId] = rho_i;
}

void CudaPBDSPH::computeDensity()
{
    dim3 pDims = int(ceil(pos_arr_.size() / BLOCK_SIZE + 0.5f));
    K_ComputeDensity << < pDims, BLOCK_SIZE >> > (rho_arr_, pos_arr_, neighbors_arr_);
}

__device__ float VisWeight(const float r, const float h)
{
    float q = r / h;
    if (q > 1.0f) return 0.0f;
    else {
        const float d = 1.0f - q;
        const float RR = h*h;
        return 45.0f / (13.0f * (float)PI * RR *h) *d;
    }
}

__global__ void K_ApplyViscosity(CudaArray<float3> velNew, CudaArray<float3> velOld, CudaArray<float3> posArr, CudaArray<float3> velArr, CudaArray<NeighborList> neighbors, float dt)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= posArr.size()) return;

    float r;
    float3 dv_i = make_float3(0.0f, 0.0f, 0.0f);
    float3 pos_i = posArr[pId];
    float3 vel_i = velArr[pId];
    float viscosity = PARAMS.viscosity / PARAMS.samplingDistance;
    float totalWeight = 0.0f;
    int nbSize = neighbors[pId].size();
    for (int ne = 0; ne < nbSize; ne++)
    {
        int j = neighbors[pId][ne];
        r = length(pos_i - posArr[j]);

        if (r > EPSILON)
        {
            float weight = VisWeight(r, PARAMS.smoothingLength);
            totalWeight += weight;
            dv_i += weight * velArr[j];
        }

    }

    float b = dt*viscosity;

    if (totalWeight < EPSILON)
        b = 0.0f;

    totalWeight = totalWeight < EPSILON ? 1.0f : totalWeight;

    dv_i /= totalWeight;


    velNew[pId] = velOld[pId] / (1.0f + b) + dv_i*b / (1.0f + b);
}

__global__ void K_UpdateVelocity(CudaArray<float3> velArr, CudaArray<float3> dVel)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= velArr.size()) return;

    velArr[pId] = dVel[pId];
}

void CudaPBDSPH::applyViscosity(float dt)
{
    dim3 pDims = int(ceil(pos_arr_.size() / BLOCK_SIZE + 0.5f));

    cudaCheck(cudaMemcpy(old_vel_.data(), vel_arr_.data(), params_.pNum * sizeof(float3), cudaMemcpyDeviceToDevice));

    for (int t = 0; t < 5; t++)
    {
        vel_arr_.swap(buffer_);
        K_ApplyViscosity << < pDims, BLOCK_SIZE >> > (vel_arr_, old_vel_, pos_arr_, buffer_, neighbors_arr_, dt);
    }

}

}//end of namespace Physika