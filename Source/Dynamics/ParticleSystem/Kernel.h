/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2018-07-03
 * @description: Definition of smoothing kernels used in SPH methods
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-27
 * @description: poslish code
 * @version    : 1.1
 */

#pragma once

#include "Core/Platform.h"
#include "Core/Utility.h"
namespace PhysIKA {
template <typename Real>
class Kernel
{
public:
    COMM_FUNC Kernel(){};
    COMM_FUNC ~Kernel(){};

    COMM_FUNC inline virtual Real Weight(const Real r, const Real h)
    {
        return Real(0);
    }

    COMM_FUNC inline virtual Real Gradient(const Real r, const Real h)
    {
        return Real(0);
    }

public:
    Real m_scale = Real(1);
};

//spiky kernel
template <typename Real>
class SpikyKernel : public Kernel<Real>
{
public:
    COMM_FUNC SpikyKernel()
        : Kernel<Real>(){};
    COMM_FUNC ~SpikyKernel(){};

    COMM_FUNC inline Real Weight(const Real r, const Real h) override
    {
        const Real q = r / h;
        if (q > 1.0f)
            return 0.0f;
        else
        {
            const Real d  = 1.0 - q;
            const Real hh = h * h;
            return 15.0f / (( Real )M_PI * hh * h) * d * d * d * this->m_scale;
        }
    }

    COMM_FUNC inline Real Gradient(const Real r, const Real h) override
    {
        const Real q = r / h;
        if (q > 1.0f)
            return 0.0;
        //else if (r==0.0f) return 0.0f;
        else
        {
            const Real d  = 1.0 - q;
            const Real hh = h * h;
            return -45.0f / (( Real )M_PI * hh * h) * d * d * this->m_scale;
        }
    }
};

template <typename Real>
class ConstantKernel : public Kernel<Real>
{
public:
    COMM_FUNC ConstantKernel()
        : Kernel<Real>(){};
    COMM_FUNC ~ConstantKernel(){};

    COMM_FUNC inline Real Weight(const Real r, const Real h) override
    {
        return Real(1);
    }

    COMM_FUNC inline Real Gradient(const Real r, const Real h) override
    {
        return Real(0);
    }
};

template <typename Real>
class SmoothKernel : public Kernel<Real>
{
public:
    COMM_FUNC SmoothKernel()
        : Kernel<Real>(){};
    COMM_FUNC ~SmoothKernel(){};

    COMM_FUNC inline Real Weight(const Real r, const Real h) override
    {
        const Real q = r / h;
        if (q > 1.0f)
            return 0.0f;
        else
        {
            return (1.0f - q * q) * this->m_scale;
        }
    }

    COMM_FUNC inline Real Gradient(const Real r, const Real h) override
    {
        const Real q = r / h;
        if (q > 1.0f)
            return 0.0f;
        else
        {
            const Real hh    = h * h;
            const Real dd    = 1 - q * q;
            const Real alpha = 1.0f;  // (Real) 945.0f / (32.0f * (Real)M_PI * hh *h);
            return -alpha * dd * this->m_scale;
        }
    }
};

//spiky kernel
template <typename Real>
class CorrectedKernel : public Kernel<Real>
{
public:
    COMM_FUNC CorrectedKernel()
        : Kernel<Real>(){};
    COMM_FUNC ~CorrectedKernel(){};

    COMM_FUNC inline Real Weight(const Real r, const Real h) override
    {
        const Real         q = r / h;
        SmoothKernel<Real> kernSmooth;
        return q * q * q * kernSmooth.Weight(r, h);

        //             const Real q = r / h;
        //             if (q > 1.0f) return 0.0f;
        //             else {
        //                 const Real d = 1.0f - q;
        //                 const Real hh = h*h;
        //                 return (1.0 - pow(q, Real(4)));
        //             }
    }

    COMM_FUNC inline Real WeightR(const Real r, const Real h)
    {
        const Real         q = r / h;
        SmoothKernel<Real> kernSmooth;
        return q * q * kernSmooth.Weight(r, h) / h;

        //             Real w = Weight(r, h);
        //             const Real q = r / h;
        //             if (q < 0.4f)
        //             {
        //                 return w / (0.4f*h);
        //             }
        //             return w / r;
    }

    COMM_FUNC inline Real WeightRR(const Real r, const Real h)
    {
        const Real         q = r / h;
        SmoothKernel<Real> kernSmooth;
        return q * kernSmooth.Weight(r, h) / (h * h);

        //             Real w = Weight(r, h);
        //             const Real q = r / h;
        //             if (q < 0.4f)
        //             {
        //                 return w / (0.16f*h*h);
        //             }
        //             return w / r / r;
    }
};

//cubic kernel
template <typename Real>
class CubicKernel : public Kernel<Real>
{
public:
    COMM_FUNC CubicKernel()
        : Kernel<Real>(){};
    COMM_FUNC ~CubicKernel(){};

    COMM_FUNC inline Real Weight(const Real r, const Real h) override
    {
        const Real hh = h * h;
        const Real q  = 2.0f * r / h;

        const Real alpha = 3.0f / (2.0f * ( Real )M_PI * hh * h);

        if (q > 2.0f)
            return 0.0f;
        else if (q >= 1.0f)
        {
            //1/6*(2-q)*(2-q)*(2-q)
            const Real d = 2.0f - q;
            return alpha / 6.0f * d * d * d;
        }
        else
        {
            //(2/3)-q*q+0.5f*q*q*q
            const Real qq  = q * q;
            const Real qqq = qq * q;
            return alpha * (2.0f / 3.0f - qq + 0.5f * qqq);
        }
    }

    COMM_FUNC inline Real Gradient(const Real r, const Real h) override
    {
        const Real hh = h * h;
        const Real q  = 2.0f * r / h;

        const Real alpha = 3.0f / (2.0f * ( Real )M_PI * hh * h);

        if (q > 2.0f)
            return 0.0f;
        else if (q >= 1.0f)
        {
            //-0.5*(2.0-q)*(2.0-q)
            const Real d = 2.0f - q;
            return -0.5f * alpha * d * d;
        }
        else
        {
            //-2q+1.5*q*q
            const Real qq = q * q;
            return alpha * (-2.0f * q + 1.5f * qq);
            //return alpha*(-0.5);
        }
    }
};

template <typename Real>
class QuarticKernel : public Kernel<Real>
{
public:
    COMM_FUNC QuarticKernel()
        : Kernel<Real>(){};
    COMM_FUNC ~QuarticKernel(){};

    COMM_FUNC inline Real Weight(const Real r, const Real h) override
    {
        const Real hh = h * h;
        const Real q  = 2.5f * r / h;
        if (q > 2.5)
            return 0.0f;
        else if (q > 1.5f)
        {
            const Real d  = 2.5f - q;
            const Real dd = d * d;
            return 0.0255f * dd * dd / hh;
        }
        else if (q > 0.5f)
        {
            const Real d  = 2.5f - q;
            const Real t  = 1.5f - q;
            const Real dd = d * d;
            const Real tt = t * t;
            return 0.0255f * (dd * dd - 5.0f * tt * tt) / hh;
        }
        else
        {
            const Real d  = 2.5f - q;
            const Real t  = 1.5f - q;
            const Real w  = 0.5f - q;
            const Real dd = d * d;
            const Real tt = t * t;
            const Real ww = w * w;
            return 0.0255f * (dd * dd - 5.0f * tt * tt + 10.0f * ww * ww) / hh;
        }
    }

    COMM_FUNC inline Real Gradient(const Real r, const Real h) override
    {
        const Real hh = h * h;
        const Real q  = 2.5f * r / h;
        if (q > 2.5)
            return 0.0f;
        else if (q > 1.5f)
        {
            //0.102*(2.5-q)^3
            const Real d = 2.5f - q;
            return -0.102f * d * d * d / hh;
        }
        else if (q > 0.5f)
        {
            const Real d = 2.5f - q;
            const Real t = 1.5f - q;
            return -0.102f * (d * d * d - 5.0f * t * t * t) / hh;
        }
        else
        {
            const Real d = 2.5f - q;
            const Real t = 1.5f - q;
            const Real w = 0.5f - q;
            return -0.102f * (d * d * d - 5.0f * t * t * t + 10.0f * w * w * w) / hh;
        }
    }
};

template <typename TReal>
class SpikyKernel2D : public Kernel<TReal>
{
public:
    COMM_FUNC SpikyKernel2D()
        : Kernel<TReal>(){};
    COMM_FUNC ~SpikyKernel2D(){};

    COMM_FUNC inline TReal Weight(const TReal r, const TReal h) override
    {
        const TReal q = r / h;
        if (q > 1.0f)
            return 0.0f;
        else
        {
            const TReal d = 1.0 - q;
            //const TReal hh = h * h;
            return 10.0f / (( TReal )M_PI * h * h) * d * d * d * this->m_scale;
        }
    }

    COMM_FUNC inline TReal Gradient(const TReal r, const TReal h) override
    {
        const TReal q = r / h;
        if (q > 1.0f)
            return 0.0;
        //else if (r==0.0f) return 0.0f;
        else
        {
            const TReal d = 1.0 - q;
            //const TReal hh = h * h;
            return -30.0f / (( TReal )M_PI * h * h * h) * d * d * this->m_scale;
        }
    }
};
}  // namespace PhysIKA
