/*
 * @file cuda_grid_hash.h 
 * @Brief class cudaGridHash
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

#ifndef PHYSIKA_DYNAMICS_SPH_GPU_SPH_CUDA_GRID_HASH_H_
#define PHYSIKA_DYNAMICS_SPH_GPU_SPH_CUDA_GRID_HASH_H_

#include "vector_types.h"
#include "Physika_Core/Cuda_Array/cuda_array.h"

namespace Physika{

#define BLOCK_SIZE    48
#define NEIGHBOR_SIZE 30

class NeighborList
{
public:
    __host__ __device__ NeighborList():size_(0) {}
    __host__ __device__ ~NeighborList() {}

    __host__ __device__ void setSize(unsigned int size) { this->size_ = size; }
    __host__ __device__ unsigned int size() const { return this->size_; }
    __host__ __device__ int& operator[] (int i) { return particle_ids_[i]; }
    __host__ __device__ int  operator[] (int i) const { return particle_ids_[i];}

private:
    unsigned int size_;
    int particle_ids_[NEIGHBOR_SIZE];
};

////////////////////////////////////////////////////////////////////////////////////////////


#define INVALID  -1
#define BUCKETS   8
#define CAPACITY 16

class CudaGridHash
{
public:
    CudaGridHash();
    ~CudaGridHash();

    void setSpace(float _h, float3 _lo, float3 _hi);
    void constructHashTable(CudaArray<float3>& pos);

    //Note: may not be correct at extreme cases.
    void queryNeighbors(CudaArray<float3>& posArr, CudaArray<NeighborList>& neighbors, float h, float pdist, int nbMaxNum);

    void queryNeighborSlow(CudaArray<float3>& posArr, CudaArray<NeighborList>& neighbors, float h, int nbMaxNum);

    void resetCounter();
    void release();

    __device__ int getIndex(int i, int j, int k);
    __device__ int getIndex(float3 pos);
    __device__ int3 getIndex3(float3 pos);

    __device__ int getCounter(int gId);
    __device__ int getParticleId(int gId, int n);

public:
    int num;
    int nx, ny, nz;

    float ds;

    float3 lo;
    float3 hi;

    int npMax;		//maximum particle number for each cell

    int* ids;
    int* counter;
};

}//end of namespace Physika

#endif //PHYSIKA_DYNAMICS_SPH_GPU_SPH_CUDA_GRID_HASH_H_