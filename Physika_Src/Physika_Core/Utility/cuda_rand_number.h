/*
 * @file cuda_rand_number.h
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

#ifndef PHYSIKA_CORE_UTILITIES_CUDA_RAND_NUMBER_H_
#define PHYSIKA_CORE_UTILITIES_CUDA_RAND_NUMBER_H_

#include "curand_kernel.h"

namespace Physika {

class CudaRandNumber
{
public:

    
    __device__ CudaRandNumber(int seed);
    __device__ ~CudaRandNumber();
    __device__ float generate();
    

    /*
    __device__ CudaRandNumber(int seed)
    {
        // seed a random number generator 
        curand_init(seed, 0, 0, &s_);
    }

    __device__ ~CudaRandNumber(){}

    
     //Generate a float number ranging from 0 to 1.
    __device__ float generate()
    {
        return curand_uniform(&s_);
    }
    */

private:
    curandState s_;
};



}//end of namespace Physika

#endif //PHYSIKA_CORE_UTILITIES_CUDA_RAND_NUMBER_H_