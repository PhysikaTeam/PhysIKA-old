/*
* @file sph_kernel.h 
* @Basic SPH_kernel class, different kernel function used in sph
* @author Sheng Yang, Fei Zhu
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
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Weight_Functions/quadratic_weight_functions.h"

namespace Physika{

/*
 * SPHKernel: base class of all SPH kernels
 * r: distance between the two particles
 * h: smooth length of the neighbor particle
 */
template <typename Scalar, int Dim>
class SPHKernel
{
public:
    SPHKernel(){}
    virtual ~SPHKernel(){}
    virtual Scalar weight(Scalar r, Scalar h) const = 0;
    virtual Scalar gradient(Scalar r, Scalar h) const = 0;
    //scale between the kernel's support radius and the partcile's smooth length
    virtual Scalar lengthScale() const = 0; 
};

/*
 * JohnsonQuadraticKernel: 
 * reference: "SPH for high velocity impact computations"
 * f(r) = a*(3/16*(r/h)^2-3/4*(r/h)+3/4) (0 <= r <= 2*h)
 * where 'a' depends on the dimension and radius of support domain
 */
template <typename Scalar, int Dim>
class JohnsonQuadraticKernel: public SPHKernel<Scalar,Dim>
{
public:
    JohnsonQuadraticKernel(){}
    ~JohnsonQuadraticKernel(){}
    inline Scalar weight(Scalar r, Scalar h) const { return kernel_.weight(r,h*lengthScale()); }
    inline Scalar gradient(Scalar r, Scalar h) const { return kernel_.gradient(r,h*lengthScale()); }
    inline Scalar lengthScale() const { return 2; }
protected:
    JohnsonQuadraticWeightFunction<Scalar,Dim> kernel_;
};



///////////////////////////////////////////////// TO DO: code below should be deleted finally ////////////////////////////////////////////////

template <typename Scalar>
class SPH_Kernel
{
public:
    virtual Scalar weight(const Scalar r, const Scalar h) { std::cout << "Weight function undefined!" << std::endl; return 0.0f; }
    virtual Scalar gradient(const Scalar r, const Scalar h) { std::cout << "Gradient function undefined!" << std::endl; return 0.0f; }
};

//spiky kernel
template <typename Scalar>
class StandardKernel : public SPH_Kernel<Scalar>
{
    virtual Scalar weight(const Scalar r, const Scalar h) 
    {
        const Scalar hh = h * h;
        const Scalar qq = r * r / hh;

        if (qq > 1)
            return 0;
        else
        {
            const Scalar dd = 1.0f - qq;
            return 315.0f / (64.0f * (Scalar)PI * hh * h) * dd * dd * dd;
        }
    }
};

template <typename Scalar>
class SmoothKernel : public SPH_Kernel<Scalar>
{
    virtual Scalar weight(const Scalar r, const Scalar h) 
    {
        const Scalar q = r / h;
        if (q > 1.0f) return 0.0f;
        else {
            return 1.0f - q * q;
        }
    }

    virtual Scalar gradient(const Scalar r, const Scalar h)
    {
        const Scalar q = r / h;
        if (q > 1.0f || r==0.0f) return 0.0f;
        else {
            const Scalar hh = h * h;
            const Scalar dd = 1 - q * q;
            const Scalar alpha = 1.0f;//(Scalar) 945.0f / (32.0f * (Scalar)M_PI * hh *h);
            return -alpha * dd;
        }
    }
};

    //spiky kernel
template <typename Scalar>
class SpikyKernel : public SPH_Kernel<Scalar>
{
    virtual Scalar weight(const Scalar r, const Scalar h) 
    {
        const Scalar q = r/h;
        if (q>1.0f) return 0.0f;
        else {
            const Scalar d = 1.0f-q;
            const Scalar hh = h*h;
            return 15.0f/((Scalar)PI * hh * h) * d * d * d;
        }
    }

    virtual Scalar gradient(const Scalar r, const Scalar h)
    {
        const Scalar q = r/h;
        if (q>1.0f) return 0.0f;
        //else if (r==0.0f) return 0.0f;
        else {
            const Scalar d = 1.0f-q;
            const Scalar hh = h*h;
            return -45.0f / ((Scalar)PI * hh*h) *d*d;
        }
    }
};


    //viscosity kernel
template <typename Scalar>
class LaplacianKernel : public SPH_Kernel<Scalar>
{
    virtual Scalar weight(const Scalar r, const Scalar h) 
    {
        const Scalar q = r/h;
        if (q>1.0f) return 0.0f;
        else {
            const Scalar d = 1.0f-q;
            const Scalar RR = h*h;
            return 45.0f/(13.0f * (Scalar)PI * RR *h) *d;
        }
    }

};


//cubic kernel
template <typename Scalar>
class CubicKernel : public SPH_Kernel<Scalar>
{
    virtual Scalar weight(const Scalar r, const Scalar h) 
    {
        const Scalar hh = h*h;
        const Scalar q = r/h;

        const Scalar alpha = 3.0f / (2.0f * (Scalar)PI * hh * h);

        if (q>2.0f) return 0.0f;
        else if (q >= 1.0f)
        {
            //1/6*(2-q)*(2-q)*(2-q)
            const Scalar d = 2.0f-q;
            return alpha/6.0f*d*d*d;
        }
        else
        {
            //(2/3)-q*q+0.5f*q*q*q
            const Scalar qq = q*q;
            const Scalar qqq = qq*q;
            return alpha*(2.0f/3.0f-qq+0.5f*qqq);
        }
    }

    virtual Scalar gradient(const Scalar r, const Scalar h)
    {
        const Scalar hh = h*h;
        const Scalar q = r/h;

        const Scalar alpha = 3.0f / (2.0f * (Scalar)PI * hh * h);

        if (q>2.0f) return 0.0f;
        else if (q >= 1.0f)
        {
            //-0.5*(2.0-q)*(2.0-q)
            const Scalar d = 2.0f-q;
            return -0.5f*alpha*d*d;
        }
        else
        {
            //-2q+1.5*q*q
            const Scalar qq = q*q;
            return alpha*(-2.0f*q+1.5f*qq);
        }
    }
};

//quadratic kernel
template <typename Scalar>
class QuadraticKernel : public SPH_Kernel<Scalar>
{
    virtual Scalar weight(const Scalar r, const Scalar h) 
    {
        const Scalar q = r/h;
        if (q>1.0f) return 0.0f;
        else {
            const Scalar alpha = 15.0f / (2.0f * (Scalar)PI);
            return alpha*(1.0f-q)*(1.0f-q);
        }
    }

    virtual Scalar gradient(const Scalar r, const Scalar h)
    {
        const Scalar q = r/h;
        if (q>1.0f) return 0.0f;
        else {
            const Scalar alpha = 15.0f / ((Scalar)PI);
            return -alpha*(1.0f-q);
        }
    }
};
template <typename Scalar>
class QuarticKernel : public SPH_Kernel<Scalar>
{
    virtual Scalar weight(const Scalar r, const Scalar h) 
    {
        const Scalar hh = h*h;
        const Scalar q = 2.5f*r/h;
        if (q>2.5) return 0.0f;
        else if (q>1.5f)
        {
            const Scalar d = 2.5f-q;
            const Scalar dd = d*d;
            return 0.0255f*dd*dd/hh;
        }
        else if (q>0.5f)
        {
            const Scalar d = 2.5f-q;
            const Scalar t = 1.5f-q;
            const Scalar dd = d*d;
            const Scalar tt = t*t;
            return 0.0255f*(dd*dd-5.0f*tt*tt)/hh;
        }
        else
        {
            const Scalar d = 2.5f-q;
            const Scalar t = 1.5f-q;
            const Scalar w = 0.5f-q;
            const Scalar dd = d*d;
            const Scalar tt = t*t;
            const Scalar ww = w*w;
            return 0.0255f*(dd*dd-5.0f*tt*tt+10.0f*ww*ww)/hh;
        }
    }

    virtual Scalar gradient(const Scalar r, const Scalar h)
    {
        const Scalar hh = h*h;
        const Scalar q = 2.5f*r/h;
        if (q>2.5) return 0.0f;
        else if (q>1.5f)
        {
            //0.102*(2.5-q)^3
            const Scalar d = 2.5f-q;
            return -0.102f*d*d*d/hh;
        }
        else if (q>0.5f)
        {
            const Scalar d = 2.5f-q;
            const Scalar t = 1.5f-q;
            return -0.102f*(d*d*d-5.0f*t*t*t)/hh;
        }
        else
        {
            const Scalar d = 2.5f-q;
            const Scalar t = 1.5f-q;
            const Scalar w = 0.5f-q;
            return -0.102f*(d*d*d-5.0f*t*t*t+10.0f*w*w*w)/hh;
        }
    }
};

template <typename Scalar>
class GaussKernel : public SPH_Kernel<Scalar>
{
    virtual Scalar weight(const Scalar r, const Scalar h) 
    {
        const double q = r/h;
        return (Scalar)pow(E, -q);
    }

    virtual Scalar gradient(const Scalar r, const Scalar h)
    {
        const double q = r/h;
        return (Scalar)-pow(E, -q);
    }
};

template <typename Scalar>
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

    static SPH_Kernel<Scalar>& createKernel(KernelType type)
    {
        SPH_Kernel<Scalar>* kern = NULL;
        switch (type)
        {
        case Spiky:
            kern = new SpikyKernel<Scalar>();
            break;
        case CubicSpline:
            kern = new CubicKernel<Scalar>();
            break;
        case QuarticSpline:
            kern = new QuarticKernel<Scalar>();
            break;
        case Smooth:
            kern = new SmoothKernel<Scalar>();
            break;
        case Standard:
            kern = new StandardKernel<Scalar>();
            break;
        case Laplacian:
            kern = new LaplacianKernel<Scalar>();
            break;
        case Quartic:
            kern = new QuarticKernel<Scalar>();
            break;
        case Gauss:
            kern = new GaussKernel<Scalar>();
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
