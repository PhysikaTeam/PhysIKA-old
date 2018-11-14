/*
 * @file kernel.h
 * @Brief various Kernel class for GPU
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


#ifndef PHYSIKA_DYNAMICS_SPH_GPU_SPH_KERNELS_H_
#define PHYSIKA_DYNAMICS_SPH_GPU_SPH_KERNELS_H_

#include <cuda_runtime.h>
#include "Physika_Core/Utilities/math_utilities.h"

namespace Physika{

/*
Various types of Kernels.
Need further consideration.
*/

class SpikyKernel
{
public:
    __host__ __device__  SpikyKernel() {};
    __host__ __device__ ~SpikyKernel() {};

    __host__ __device__ inline float weight(const float r, const float h)
    {
        const float q = r / h;
        if (q > 1.0f) 
            return 0.0f;
        else 
        {
            const float d = 1.0f - q;
            const float hh = h*h;
            return 15.0f / ((float)PI * hh * h) * d * d * d;
        }
    }

    __host__ __device__ inline float gradient(const float r, const float h)
    {
        const float q = r / h;
        if (q > 1.0f) 
            return 0.0f;
        //else if (r==0.0f) return 0.0f;
        else 
        {
            const float d = 1.0f - q;
            const float hh = h*h;
            return -45.0f / ((float)PI * hh*h) *d*d;
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

class GaussKernel
{
public:
    __host__ __device__  GaussKernel() {};
    __host__ __device__ ~GaussKernel() {};

    __host__ __device__ inline float weight(const float r, const float h)
    {
        const float q = r / h;
        if (q > 1.0f) 
            return 0.0f;
        else 
        {
            const float d = pow((float)E, -q*q);
            return d;
        }
    }

    __host__ __device__ inline float gradient(const float r, const float h)
    {
        const float q = r / h;
        if (q > 1.0f) 
            return 0.0f;
        else 
        {
            const float d = pow((float)E, -q*q);
            return -d;
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

class CubicKernel
{
public:
    __host__ __device__  CubicKernel() {}
    __host__ __device__ ~CubicKernel() {}

    __host__ __device__ inline float weight(const float r, const float h)
    {
        const float hh = h*h;
        const float q = 2.0f*r / h;

        const float alpha = 3.0f / (2.0f * (float)PI * hh * h);

        if (q > 2.0f) 
            return 0.0f;
        else if (q >= 1.0f)
        {
            //1/6*(2-q)*(2-q)*(2-q)
            const float d = 2.0f - q;
            return alpha / 6.0f*d*d*d;
        }
        else
        {
            //(2/3)-q*q+0.5f*q*q*q
            const float qq = q*q;
            const float qqq = qq*q;
            return alpha*(2.0f / 3.0f - qq + 0.5f*qqq);
        }
    }

    __host__ __device__ inline float gradient(const float r, const float h)
    {
        const float hh = h*h;
        const float q = 2.0f*r / h;

        const float alpha = 3.0f / (2.0f * (float)PI * hh * h);

        if (q > 2.0f) 
            return 0.0f;
        else if (q >= 1.0f)
        {
            //-0.5*(2.0-q)*(2.0-q)
            const float d = 2.0f - q;
            return -0.5f*alpha*d*d;
        }
        else
        {
            //-2q+1.5*q*q
            const float qq = q*q;
            return alpha*(-2.0f*q + 1.5f*qq);
            //return alpha*(-0.5);
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

class SmoothKernel
{
public:
    __host__ __device__  SmoothKernel() {}
    __host__ __device__ ~SmoothKernel() {}

    __host__ __device__ inline float weight(const float r, const float h)
    {
        const float q = r / h;
        if (q > 1.0f) 
            return 0.0f;
        else 
            return 1.0f - q*q;
    }

    __host__ __device__ inline float gradient(const float r, const float h)
    {
        const float q = r / h;
        if (q > 1.0f) 
            return 0.0f;
        else 
        {
            const float hh = h*h;
            const float dd = 1 - q*q;
            const float alpha = 1.0f; //(float) 945.0f / (32.0f * (float)M_PI * hh *h);
            return -alpha * dd;
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

class QuarticKernel
{
public:
    __host__ __device__  QuarticKernel() {}
    __host__ __device__ ~QuarticKernel() {}

    __host__ __device__ inline float weight(const float r, const float h)
    {
        const float hh = h*h;
        const float q = 2.5f*r / h;

        if (q > 2.5) 
            return 0.0f;
        else if (q > 1.5f)
        {
            const float d = 2.5f - q;
            const float dd = d*d;
            return 0.0255f*dd*dd / hh;
        }
        else if (q > 0.5f)
        {
            const float d = 2.5f - q;
            const float t = 1.5f - q;
            const float dd = d*d;
            const float tt = t*t;
            return 0.0255f*(dd*dd - 5.0f*tt*tt) / hh;
        }
        else
        {
            const float d = 2.5f - q;
            const float t = 1.5f - q;
            const float w = 0.5f - q;
            const float dd = d*d;
            const float tt = t*t;
            const float ww = w*w;
            return 0.0255f*(dd*dd - 5.0f*tt*tt + 10.0f*ww*ww) / hh;
        }
    }

    __host__ __device__ inline float gradient(const float r, const float h)
    {
        const float hh = h*h;
        const float q = 2.5f*r / h;
        if (q > 2.5) 
            return 0.0f;
        else if (q > 1.5f)
        {
            //0.102*(2.5-q)^3
            const float d = 2.5f - q;
            return -0.102f*d*d*d / hh;
        }
        else if (q > 0.5f)
        {
            const float d = 2.5f - q;
            const float t = 1.5f - q;
            return -0.102f*(d*d*d - 5.0f*t*t*t) / hh;
        }
        else
        {
            const float d = 2.5f - q;
            const float t = 1.5f - q;
            const float w = 0.5f - q;
            return -0.102f*(d*d*d - 5.0f*t*t*t + 10.0f*w*w*w) / hh;
        }
    }
};

}//end of namespace Physika

#endif //PHYSIKA_DYNAMICS_SPH_GPU_SPH_KERNELS_H_