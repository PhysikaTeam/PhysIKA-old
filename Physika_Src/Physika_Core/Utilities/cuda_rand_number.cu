/*
 * @file cuda_rand_number.cu
 * @Brief class CudaRandNumber, generate rand number for GPU
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

#include "Physika_Core/Utilities/cuda_rand_number.h"

namespace Physika {


__device__ CudaRandNumber::CudaRandNumber(int seed)
{
    // seed a random number generator 
    curand_init(seed, 0, 0, &s_);
}

__device__ CudaRandNumber::~CudaRandNumber()
{

}

//brief	Generate a float number ranging from 0 to 1.
__device__ float CudaRandNumber::generate()
{
    return curand_uniform(&s_);
}


}//end of namespace Physika

