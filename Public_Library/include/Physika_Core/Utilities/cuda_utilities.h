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

#define CPU_GPU_FUNC_DECL __host__ __device__
#define CPU_FUNC_DECL __host__
#define GPU_FUNC_DECL __device__

namespace Physika{

#define cudaCheck(x) { cudaError_t err = x; if (err != cudaSuccess) { printf("Cuda error: %d in %s at %s:%d\n", err, #x, __FILE__, __LINE__); printf("Cuda status: %s\n", cudaGetErrorString( cudaGetLastError() ) ); assert(0);} }

}// end of namespace Physika

#endif //PHYSIKA_CORE_UTILITIES_CUDA_UTILITIES_H_