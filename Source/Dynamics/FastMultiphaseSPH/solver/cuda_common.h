
#pragma once

#include <cuda_runtime.h>
#include "../utility/helper_cuda.h"
#include "../math/geometry.h"

#define GRID_UNDEF 99999999

inline uint iDivUp(uint a, uint b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
inline void computeBlockSize(uint n, uint blockSize, uint& numBlocks, uint& numThreads)
{
    numThreads = fmin(blockSize, n);
    numBlocks  = iDivUp(n, numThreads);
}

__device__ __forceinline__ cint3 calcGridPos(
    cfloat3 p,
    cfloat3 xmin,
    float   dx)
{
    cint3 gridPos;
    gridPos.x = floorf((p.x - xmin.x) / dx);
    gridPos.y = floorf((p.y - xmin.y) / dx);
    gridPos.z = floorf((p.z - xmin.z) / dx);
    return gridPos;
}

__device__ __forceinline__ uint calcGridHash(
    cint3 cell_indices,
    cint3 res)
{
    if (cell_indices.x < 0 || cell_indices.x >= res.x || cell_indices.y < 0 || cell_indices.y >= res.y || cell_indices.z < 0 || cell_indices.z >= res.z)
        return GRID_UNDEF;

    return cell_indices.y * res.x * res.z
           + cell_indices.z * res.x
           + cell_indices.x;
}