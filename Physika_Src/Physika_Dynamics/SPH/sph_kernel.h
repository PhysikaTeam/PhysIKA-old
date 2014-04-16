/*
* @file sph_kernel.h 
* @Basic SPH_kernel class, different kernel function used in sph
* @author Sheng Yang
* 
* This file is part of Physika, a versatile physics simulation library.
* Copyright (C) 2013 Physika Group.
*
* This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
* If a copy of the GPL was not distributed with this file, you can obtain one at:
* http://www.gnu.org/licenses/gpl-2.0.html
*
*/

#ifndef PHYSIKA_DYNAMICS_SPH_SPH_KERNEL_H_
#define PHYSIKA_DYNAMICS_SPH_SPH_KERNEL_H_

#include <cmath>
#include <string>
#include <iostream>
//#include <math.h>
#define M_PI 3.14159265358979323846
#define M_E 2.71828182845904523536
namespace Physika{



    class SPH_Kernel
    {
    public:
        virtual float weight(const float r, const float h) { std::cout << "Weight function undefined!" << std::endl; return 0.0f; }
        virtual float gradient(const float r, const float h) { std::cout << "Gradient function undefined!" << std::endl; return 0.0f; }
    };

    //spiky kernel
    class StandardKernel : public SPH_Kernel
    {
        virtual float weight(const float r, const float h) 
        {
            const float hh = h*h;
            const float qq = r*r/hh;

            if (qq > 1)
                return 0;
            else
            {
                const float dd = 1.0f - qq;
                return 315.0f/(64.0f*(float)M_PI*hh*h)*dd*dd*dd;
            }
        }
    };

    class SmoothKernel : public SPH_Kernel
    {
        virtual float weight(const float r, const float h) 
        {
            const float q = r/h;
            if (q>1.0f) return 0.0f;
            else {
                return 1.0f-q*q;
            }
        }

        virtual float gradient(const float r, const float h)
        {
            const float q = r/h;
            if (q>1.0f || r==0.0f) return 0.0f;
            else {
                const float hh = h*h;
                const float dd = 1-q*q;
                const float alpha = 1.0f;//(float) 945.0f / (32.0f * (float)M_PI * hh *h);
                return -alpha * dd;
            }
        }
    };

    //spiky kernel
    class SpikyKernel : public SPH_Kernel
    {
        virtual float weight(const float r, const float h) 
        {
            const float q = r/h;
            if (q>1.0f) return 0.0f;
            else {
                const float d = 1.0f-q;
                const float hh = h*h;
                return 15.0f/((float)M_PI * hh * h) * d * d * d;
            }
        }

        virtual float gradient(const float r, const float h)
        {
            const float q = r/h;
            if (q>1.0f) return 0.0f;
            //else if (r==0.0f) return 0.0f;
            else {
                const float d = 1.0f-q;
                const float hh = h*h;
                return -45.0f / ((float)M_PI * hh*h) *d*d;
            }
        }
    };


    //viscosity kernel
    class LaplacianKernel : public SPH_Kernel
    {
        virtual float weight(const float r, const float h) 
        {
            const float q = r/h;
            if (q>1.0f) return 0.0f;
            else {
                const float d = 1.0f-q;
                const float RR = h*h;
                return 45.0f/(13.0f * (float)M_PI * RR *h) *d;
            }
        }

    };


    //cubic kernel
    class CubicKernel : public SPH_Kernel
    {
        virtual float weight(const float r, const float h) 
        {
            const float hh = h*h;
            const float q = r/h;

            const float alpha = 3.0f / (2.0f * (float)M_PI * hh * h);

            if (q>2.0f) return 0.0f;
            else if (q >= 1.0f)
            {
                //1/6*(2-q)*(2-q)*(2-q)
                const float d = 2.0f-q;
                return alpha/6.0f*d*d*d;
            }
            else
            {
                //(2/3)-q*q+0.5f*q*q*q
                const float qq = q*q;
                const float qqq = qq*q;
                return alpha*(2.0f/3.0f-qq+0.5f*qqq);
            }
        }

        virtual float gradient(const float r, const float h)
        {
            const float hh = h*h;
            const float q = r/h;

            const float alpha = 3.0f / (2.0f * (float)M_PI * hh * h);

            if (q>2.0f) return 0.0f;
            else if (q >= 1.0f)
            {
                //-0.5*(2.0-q)*(2.0-q)
                const float d = 2.0f-q;
                return -0.5f*alpha*d*d;
            }
            else
            {
                //-2q+1.5*q*q
                const float qq = q*q;
                return alpha*(-2.0f*q+1.5f*qq);
            }
        }
    };

    //quadratic kernel
    class QuadraticKernel : public SPH_Kernel
    {
        virtual float weight(const float r, const float h) 
        {
            const float q = r/h;
            if (q>1.0f) return 0.0f;
            else {
                const float alpha = 15.0f / (2.0f * (float)M_PI);
                return alpha*(1.0f-q)*(1.0f-q);
            }
        }

        virtual float gradient(const float r, const float h)
        {
            const float q = r/h;
            if (q>1.0f) return 0.0f;
            else {
                const float alpha = 15.0f / ((float)M_PI);
                return -alpha*(1.0f-q);
            }
        }
    };

    class QuarticKernel : public SPH_Kernel
    {
        virtual float weight(const float r, const float h) 
        {
            const float hh = h*h;
            const float q = 2.5f*r/h;
            if (q>2.5) return 0.0f;
            else if (q>1.5f)
            {
                const float d = 2.5f-q;
                const float dd = d*d;
                return 0.0255f*dd*dd/hh;
            }
            else if (q>0.5f)
            {
                const float d = 2.5f-q;
                const float t = 1.5f-q;
                const float dd = d*d;
                const float tt = t*t;
                return 0.0255f*(dd*dd-5.0f*tt*tt)/hh;
            }
            else
            {
                const float d = 2.5f-q;
                const float t = 1.5f-q;
                const float w = 0.5f-q;
                const float dd = d*d;
                const float tt = t*t;
                const float ww = w*w;
                return 0.0255f*(dd*dd-5.0f*tt*tt+10.0f*ww*ww)/hh;
            }
        }

        virtual float gradient(const float r, const float h)
        {
            const float hh = h*h;
            const float q = 2.5f*r/h;
            if (q>2.5) return 0.0f;
            else if (q>1.5f)
            {
                //0.102*(2.5-q)^3
                const float d = 2.5f-q;
                return -0.102f*d*d*d/hh;
            }
            else if (q>0.5f)
            {
                const float d = 2.5f-q;
                const float t = 1.5f-q;
                return -0.102f*(d*d*d-5.0f*t*t*t)/hh;
            }
            else
            {
                const float d = 2.5f-q;
                const float t = 1.5f-q;
                const float w = 0.5f-q;
                return -0.102f*(d*d*d-5.0f*t*t*t+10.0f*w*w*w)/hh;
            }
        }
    };

    class GaussKernel : public SPH_Kernel
    {
        virtual float weight(const float r, const float h) 
        {
            const double q = r/h;
            return (float)pow(M_E, -q);
        }

        virtual float gradient(const float r, const float h)
        {
            const double q = r/h;
            return (float)-pow(M_E, -q);
        }
    };

    class KernelFactory
    {
    public:

        enum KernelType
        {
            Spiky,
            CubicSpline,
            QuarticSpline,
            Smooth,
            Standard,
            Laplacian,
            Quartic,
            Gauss,
        };

        static SPH_Kernel& createKernel(KernelType type)
        {
            SPH_Kernel* kern = NULL;
            switch (type)
            {
            case Spiky:
                kern = new SpikyKernel;
                break;
            case CubicSpline:
                kern = new CubicKernel;
                break;
            case QuarticSpline:
                kern = new QuarticKernel;
                break;
            case Smooth:
                kern = new SmoothKernel;
                break;
            case Standard:
                kern = new StandardKernel;
                break;
            case Laplacian:
                kern = new LaplacianKernel;
                break;
            case Quartic:
                kern = new QuarticKernel;
                break;
            case Gauss:
                kern = new GaussKernel;
                break;
            default:
                kern = NULL;
                break;
            }

            return *kern;
        }
    };
}
//end of namespace Physika

#endif //PHYSIKA_DYNAMICS_SPH_SPH_KERNEL_H_
