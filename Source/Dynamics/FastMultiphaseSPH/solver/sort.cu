

#include "cuda_common.h"
#include "../utility/helper_cuda.h"
#include "sort.h"
#include <thrust\device_vector.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

__global__ void computeParticleHash(
    cfloat3* pos,
    uint*    particle_hash,
    uint*    particle_index,
    cfloat3  xmin,
    float    dx,
    cint3    res,
    int      num_particles)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles)
        return;

    auto x       = pos[i];
    auto gridPos = calcGridPos(x, xmin, dx);
    auto hash    = calcGridHash(gridPos, res);

    particle_hash[i]  = hash;
    particle_index[i] = i;
}

void computeParticleHash_host(
    cfloat3* pos,
    uint*    particle_hash,
    uint*    particle_index,
    cfloat3  xmin,
    float    dx,
    cint3    res,
    int      num_particles)
{

    uint numBlocks, numThreads;
    computeBlockSize(num_particles, 256, numBlocks, numThreads);

    getLastCudaError("Kernel execution failed: before compute particle hash");
    computeParticleHash<<<numBlocks, numThreads>>>(
        pos,
        particle_hash,
        particle_index,
        xmin,
        dx,
        res,
        num_particles);

    cudaThreadSynchronize();
    getLastCudaError("Kernel execution failed: compute particle hash");
}

void sortParticleHash(
    uint* particle_hash,
    uint* particle_index,
    int   num_particles)
{
    thrust::sort_by_key(
        thrust::device_ptr<uint>(particle_hash),
        thrust::device_ptr<uint>(particle_hash + num_particles),
        thrust::device_ptr<uint>(particle_index));
}

__global__ void findCellStart(
    uint* particle_hash,
    uint* grid_cell_start,
    uint* grid_cell_end,
    int   num_particles)
{
    uint                   index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    extern __shared__ uint sharedHash[];
    uint                   hash;

    if (index < num_particles)
    {
        hash                        = particle_hash[index];
        sharedHash[threadIdx.x + 1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            sharedHash[0] = particle_hash[index - 1];
        }
    }
    __syncthreads();

    if (index < num_particles)
    {

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            if (hash != GRID_UNDEF)
                grid_cell_start[hash] = index;

            if (index > 0)
                grid_cell_end[sharedHash[threadIdx.x]] = index;
        }
        if (index == num_particles - 1)
        {
            if (hash != GRID_UNDEF)
                grid_cell_end[hash] = index + 1;
        }
    }
}

void findCellStart_host(
    uint* particle_hash,
    uint* grid_cell_start,
    uint* grid_cell_end,
    int   num_particles,
    int   num_cells)
{
    uint numThreads, numBlocks;
    computeBlockSize(num_particles, 256, numBlocks, numThreads);
    cudaMemset(grid_cell_start, 0xffffffff, num_cells * sizeof(uint));

    //shared memory size
    uint smemSize = sizeof(uint) * (numThreads + 1);

    findCellStart<<<numBlocks, numThreads, smemSize>>>(
        particle_hash,
        grid_cell_start,
        grid_cell_end,
        num_particles);

    cudaThreadSynchronize();
    getLastCudaError("Kernel execution failed: find cell start");
}

__global__ void FindCellStart(
    uint* particle_hash,
    uint* grid_cell_start,
    uint* grid_cell_end,

    uint* sortedIndices,
    uint* indicesAfterSort,
    int   num_particles)
{
    uint                   index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    extern __shared__ uint sharedHash[];
    uint                   hash;

    if (index < num_particles)
    {
        hash                        = particle_hash[index];
        sharedHash[threadIdx.x + 1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            sharedHash[0] = particle_hash[index - 1];
        }

        // label indices after sort
        uint originalIndex              = sortedIndices[index];
        indicesAfterSort[originalIndex] = index;
    }
    __syncthreads();

    if (index < num_particles)
    {

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            if (hash != GRID_UNDEF)
                grid_cell_start[hash] = index;

            if (index > 0)
                grid_cell_end[sharedHash[threadIdx.x]] = index;
        }
        if (index == num_particles - 1)
        {
            if (hash != GRID_UNDEF)
                grid_cell_end[hash] = index + 1;
        }
    }
}

void FindCellStartHost(
    uint* particle_hash,
    uint* grid_cell_start,
    uint* grid_cell_end,

    uint* sortedIndices,
    uint* indicesAfterSort,
    int   num_particles,
    int   num_cells)
{
    uint numThreads, numBlocks;
    computeBlockSize(num_particles, 256, numBlocks, numThreads);
    cudaMemset(grid_cell_start, 0xffffffff, num_cells * sizeof(uint));

    //shared memory size
    uint smemSize = sizeof(uint) * (numThreads + 1);

    FindCellStart<<<numBlocks, numThreads, smemSize>>>(
        particle_hash,
        grid_cell_start,
        grid_cell_end,

        sortedIndices,
        indicesAfterSort,
        num_particles);

    cudaThreadSynchronize();
    getLastCudaError("Kernel execution failed: find cell start");
}