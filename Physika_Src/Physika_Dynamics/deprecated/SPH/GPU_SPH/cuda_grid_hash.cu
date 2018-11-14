/*
* @file cuda_grid_hash.cpp
* @Brief class CudaGridHash
* @author CudaGridHash
*
* This file is part of Physika, a versatile physics simulation library.
* Copyright (C) 2013- Physika Group.
*
* This Source Code Form is subject to the terms of the GNU General Public License v2.0.
* If a copy of the GPL was not distributed with this file, you can obtain one at:
* http://www.gnu.org/licenses/gpl-2.0.html
*
*/

#include "device_atomic_functions.h"

#include "Physika_Core/Timer/timer.h"
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Core/Utilities/cuda_helper_math.h"

#include "Physika_Dynamics/SPH/GPU_SPH/cuda_grid_hash.h"

namespace Physika{

__constant__ int offset[27][3] = { 0,  0,  0,
                                   0,  0,  1,
                                   0,  1,  0,
                                   1,  0,  0,
                                   0,  0, -1,
                                   0, -1,  0,
                                  -1,  0,  0,
                                   0,  1,  1,
                                   0,  1, -1,
                                   0, -1,  1,
                                   0, -1, -1,
                                   1,  0,  1,
                                   1,  0, -1,
                                  -1,  0,  1,
                                  -1,  0, -1,
                                   1,  1,  0,
                                   1, -1,  0,
                                  -1,  1,  0,
                                  -1, -1,  0,
                                   1,  1,  1,
                                   1,  1, -1,
                                   1, -1,  1,
                                  -1,  1,  1,
                                   1, -1, -1,
                                  -1,  1, -1,
                                  -1, -1,  1,
                                  -1, -1, -1
                                };


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CudaGridHash::CudaGridHash()
{

}

CudaGridHash::~CudaGridHash()
{

}

void CudaGridHash::setSpace(float _h, float3 _lo, float3 _hi)
{
    ds = _h;
    lo = _lo - _h;

    float3 nSeg = (_hi - _lo) / ds;

    nx = ceil(nSeg.x) + 1;
    ny = ceil(nSeg.y) + 1;
    nz = ceil(nSeg.z) + 1;
    hi = lo + make_float3(nx, ny, nz)*ds;

    num = nx*ny*nz;

    npMax = 32;

    cudaCheck(cudaMalloc((void**)&counter, num * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&ids, num * npMax * sizeof(int)));
}

/*******************************************************************************************************************************************/

__global__ void K_ConstructHashTable(CudaGridHash hash, CudaArray<float3> pos)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= pos.size()) return;

    int gId = hash.getIndex(pos[pId]);
    if (gId < 0) return;

    int index = atomicAdd(&(hash.counter[gId]), 1);
    index = min(index, hash.npMax - 1);

    hash.ids[gId * hash.npMax + index] = pId;
}

void CudaGridHash::constructHashTable(CudaArray<float3> & pos)
{
    dim3 pDims = int(ceil(pos.size() / BLOCK_SIZE + 0.5f));
    K_ConstructHashTable <<<pDims, BLOCK_SIZE >>> (*this, pos);
}

/*******************************************************************************************************************************************/

void CudaGridHash::resetCounter()
{
    cudaCheck(cudaMemset(counter, 0, num * sizeof(int)));
}

void CudaGridHash::release()
{
    cudaCheck(cudaFree(counter));
    cudaCheck(cudaFree(ids));
}

/*******************************************************************************************************************************************/

__global__ void K_ComputeNeighbors(CudaArray<float3> posArr, CudaArray<NeighborList> neighbors, CudaGridHash hash, float h, float pdist, float nbMaxNum)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId > posArr.size()) return;

    int tId = threadIdx.x;
    int ids[BUCKETS][CAPACITY];
    float distance[CAPACITY];
    int counter[BUCKETS];

    for (int i = 0; i < BUCKETS; i++)
        counter[i] = 0;

    float3 pos_ijk = posArr[pId];
    int3 gId3 = hash.getIndex3(pos_ijk);

    for (int c = 0; c < 27; c++)
    {
        int cId = hash.getIndex(gId3.x + offset[c][0], gId3.y + offset[c][1], gId3.z + offset[c][2]);
        if (cId >= 0) 
        {
            int totalNum = min(hash.getCounter(cId), hash.npMax);
            for (int i = 0; i < totalNum; i++) 
            {
                int nbId = hash.getParticleId(cId, i);
                float d_ij = length(pos_ijk - posArr[nbId]);
                if (d_ij < h)
                {
                    int bId = floor(pow(d_ij / h, 3.0f)*BUCKETS);
                    bId = clamp(bId, 0, BUCKETS - 1);

                    //printf("exceeded %i", bId);

                    if (counter[bId] < CAPACITY)
                    {
                        ids[bId][counter[bId]] = nbId;
                        counter[bId]++;
                    }
                    /*
                    else
                     	printf("exceeded");
                    */
                }
            }
        }
    }

    int nbSize = 0;
    int totalNum = 0;
    for (int bId = 0; bId < BUCKETS; bId++)
    {
        int btSize = counter[bId];
        totalNum += btSize;
        if (totalNum <= nbMaxNum)
        {
            for (int k = 0; k < btSize; k++)
            {
                neighbors[pId][nbSize] = ids[bId][k];
                nbSize++;
            }
        }
        else
        {
            for (int i = 0; i < btSize; i++)
                distance[i] = length(pos_ijk - posArr[ids[bId][i]]);
            
            int rN = nbMaxNum - totalNum + btSize;
            for (int k = 0; k < rN; k++)
            {
                float minDist = distance[k];
                int id = k;
                for (int t = k + 1; t < btSize; t++)
                {
                    if (distance[t] < minDist)
                    {
                        minDist = distance[t];
                        id = t;
                    }
                }
                neighbors[pId][nbSize] = ids[bId][id];
                nbSize++;
                distance[id] = distance[k];
                ids[bId][id] = ids[bId][k];
            }
        }
    }

    neighbors[pId].setSize(nbSize);
}

/*******************************************************************************************************************************************/



void CudaGridHash::queryNeighbors(CudaArray<float3>& posArr, CudaArray<NeighborList>& neighbors, float h, float pdist, int nbMaxNum)
{
    resetCounter();
    constructHashTable(posArr);

    dim3 pDims = int(ceil(posArr.size() / BLOCK_SIZE + 0.5f));

    Timer timer;
    timer.startTimer();

    //------------------------------------------------------------------------------------------
    K_ComputeNeighbors<<<pDims, BLOCK_SIZE >>>(posArr, neighbors, *this, h, pdist, nbMaxNum);
    //------------------------------------------------------------------------------------------

   timer.stopTimer();
   std::cout << "time cost for ComputeNeighbors: " << timer.getElapsedTime() << std::endl;
}

/*******************************************************************************************************************************************/

__global__ void K_ComputeNeighborSlow(CudaArray<float3> posArr, CudaArray<NeighborList> neighbors, CudaGridHash hash, float h, float nbMaxNum)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId > posArr.size()) return;

    int tId = threadIdx.x;
    int ids[NEIGHBOR_SIZE];
    float distance[NEIGHBOR_SIZE];

    float3 pos_ijk = posArr[pId];
    int3 gId3 = hash.getIndex3(pos_ijk);

    int counter = 0;
    for (int c = 0; c < 27; c++)
    {
        int cId = hash.getIndex(gId3.x + offset[c][0], gId3.y + offset[c][1], gId3.z + offset[c][2]);
        if (cId >= 0) 
        {
            int totalNum = min(hash.getCounter(cId), hash.npMax);
            for (int i = 0; i < totalNum; i++) 
            {
                int nbId = hash.getParticleId(cId, i);
                float d_ij = length(pos_ijk - posArr[nbId]);
                if (d_ij < h)
                {
                    if (counter < nbMaxNum)
                    {
                        ids[counter] = nbId;
                        distance[counter] = d_ij;
                        counter++;
                    }
                    else
                    {
                        int maxId = 0;
                        float maxDist = distance[0];
                        for (int ne = 1; ne < nbMaxNum; ne++)
                        {
                            if (maxDist < distance[ne])
                            {
                                maxDist = distance[ne];
                                maxId = ne;
                            }
                        }
                        if (d_ij < distance[maxId])
                        {
                            distance[maxId] = d_ij;
                            ids[maxId] = nbId;
                        }
                    }
                }
            }
        }
    }

   
    for (int bId = 0; bId < counter; bId++)
        neighbors[pId][bId] = ids[bId];

    neighbors[pId].setSize(counter);
}

/*******************************************************************************************************************************************/

void CudaGridHash::queryNeighborSlow(CudaArray<float3>& posArr, CudaArray<NeighborList>& neighbors, float h, int nbMaxNum)
{
    resetCounter();
    constructHashTable(posArr);

    dim3 pDims = int(ceil(posArr.size() / BLOCK_SIZE + 0.5f));

    Timer timer;
    timer.startTimer();

    //------------------------------------------------------------------------------------------
    K_ComputeNeighborSlow <<<pDims, BLOCK_SIZE>>> (posArr, neighbors, *this, h, nbMaxNum);
    //------------------------------------------------------------------------------------------

    timer.stopTimer();
    std::cout << "time cost for ComputeNeighbors: " << timer.getElapsedTime() << std::endl;
}

__device__ int CudaGridHash::getIndex(int i, int j, int k)
{
    if (i < 0 || i >= nx) return INVALID;
    if (j < 0 || j >= ny) return INVALID;
    if (k < 0 || k >= nz) return INVALID;

    return i + j*nx + k*nx*ny;
}

__device__ int CudaGridHash::getIndex(float3 pos)
{
    int i = floor((pos.x - lo.x) / ds);
    int j = floor((pos.y - lo.y) / ds);
    int k = floor((pos.z - lo.z) / ds);

    return getIndex(i, j, k);
}

__device__ int3 CudaGridHash::getIndex3(float3 pos)
{
    int i = floor((pos.x - lo.x) / ds);
    int j = floor((pos.y - lo.y) / ds);
    int k = floor((pos.z - lo.z) / ds);

    return make_int3(i, j, k);
}

__device__ int CudaGridHash::getCounter(int gId) 
{ 
    return counter[gId]; 
}

__device__ int CudaGridHash::getParticleId(int gId, int n) 
{ 
    return ids[gId*npMax + n];
}


}//end of namespace Physika