#pragma once
#include "sph_common.h"

class SPHHelper
{
private:
    float _smoothRadius;
    float _factor;
    float _gradientFactor;
    float _laplacianFactor;

public:
    HDFUNC void SetupCubic(float smoothRadius)
    {
        _smoothRadius   = smoothRadius;
        _factor         = 1.0f / 3.141593f / pow(smoothRadius, 3.0f);
        _gradientFactor = 1.5f / 3.141593f / pow(smoothRadius, 4.0f);
    }

    HDFUNC float Cubic(float d)
    {
        float q = d / _smoothRadius;
        float func;
        if (q >= 2)
            return 0;
        if (q >= 1)
            func = (2 - q) * (2 - q) * (2 - q) * 0.25f;
        else
            func = (0.666667f - q * q + 0.5f * q * q * q) * 1.5f;
        return func * _factor;
    }

    HDFUNC cfloat3 CubicGradient(cfloat3 xij)
    {
        float d = xij.length();
        float q = d / _smoothRadius;
        if (!(q < 2 && q > EPSILON))
            return cfloat3(0, 0, 0);
        float func;
        if (q >= 1)
            func = 0.5f * (2 - q) * (2 - q) * (-1);
        else
            func = (-2 * q + 1.5f * q * q);
        return xij * (func * _gradientFactor / d);
    }
};
