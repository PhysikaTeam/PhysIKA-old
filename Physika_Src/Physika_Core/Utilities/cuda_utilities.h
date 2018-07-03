/*
 * @file cuda_utilities.h
 * @Brief cuda utilities
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

#ifndef PHYSIKA_CORE_UTILITIES_CUDA_UTILITIES_H_
#define PHYSIKA_CORE_UTILITIES_CUDA_UTILITIES_H_

#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "vector_types.h"
#include "vector_functions.h"
#include "cuda_helper_math.h"
#include <iostream>

#define CPU_GPU_FUNC_DECL __host__ __device__
#define CPU_FUNC_DECL __host__
#define GPU_FUNC_DECL __device__

namespace Physika{

#define INVALID -1
#define EPSILON   1e-6
#define M_PI 3.14159265358979323846
#define M_E 2.71828182845904523536

	#define BLOCK_SIZE 64
	#define cudaCheck(x) { cudaError_t err = x; if (err != cudaSuccess) { printf("Cuda error: %d in %s at %s:%d\n", err, #x, __FILE__, __LINE__); printf("Cuda status: %s\n", cudaGetErrorString( cudaGetLastError() ) ); assert(0);} }

	using uint = unsigned int;

	static uint iDivUp(uint a, uint b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}

	// compute grid and thread block size for a given number of elements
	static uint cudaGridSize(uint totalSize, uint blockSize)
	{
		return iDivUp(totalSize, blockSize);
	}

	static dim3 cudaGridSize3D(uint3 totalSize, uint3 blockSize)
	{
		dim3 gridDims;
		gridDims.x = iDivUp(totalSize.x, blockSize.x);
		gridDims.y = iDivUp(totalSize.y, blockSize.y);
		gridDims.z = iDivUp(totalSize.z, blockSize.z);

		return gridDims;
	}

}// end of namespace Physika

#endif //PHYSIKA_CORE_UTILITIES_CUDA_UTILITIES_H_