/*
 * @file boundary.cu
 * @Brief boundary condition
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

#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Utilities/cuda_rand_number.h"
#include "Physika_Core/Utilities/cuda_helper_math.h"
#include "Physika_Core/Cuda_Array/cuda_array.h"

#include "Physika_Dynamics/SPH/GPU_SPH/boundary.h"

namespace Physika {

__global__ void K_Constrain(CudaArray<float3> posArr, CudaArray<float3> velArr, CudaDistanceField df, float normalFriction, float tangentialFriction, float dt)
{

    int pId = threadIdx.x + (blockIdx.x * blockDim.x);

    if (pId >= posArr.size()) return;

    float3 pos = posArr[pId];
    float3 vec = velArr[pId];

    float dist;
    float3 normal;
    df.getDistance(pos, dist, normal);

    // constrain particle
    if (dist <= 0)
    {
        float olddist = -dist;

        CudaRandNumber rGen(pId);
        dist = 0.0001f*rGen.generate();

        // reflect position
        pos -= (olddist + dist)*normal;

        // reflect velocity
        float vlength = length(vec);
        float vec_n = dot(vec, normal);
        float3 vec_normal = vec_n*normal;
        float3 vec_tan = vec - vec_normal;
        if (vec_n > 0) vec_normal = -vec_normal;
        vec_normal *= (1.0f - normalFriction);
        vec = vec_normal + vec_tan;
        vec *= pow(float(E), -dt*tangentialFriction);
    }

    posArr[pId] = pos;
    velArr[pId] = vec;
}

//=====================================================================================================================================================================


//=====================================================================================================================

#define BLOCK_SIZE    48

static uint iDivUp(uint a, uint b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
static uint cudaGridSize(uint totalSize, uint blockSize)
{
    return iDivUp(totalSize, blockSize);
}

//=====================================================================================================================

void BarrierCudaDistanceField::constrain(CudaArray<float3>& pos_arr, CudaArray<float3>& vel_arr, float dt) const
{
    uint pDim = cudaGridSize(pos_arr.size(), BLOCK_SIZE);
    K_Constrain <<<pDim, BLOCK_SIZE >>> (pos_arr, vel_arr, *distance_field_, normal_friction_, tangential_friction_, dt);
}

}//end of namespace Physika