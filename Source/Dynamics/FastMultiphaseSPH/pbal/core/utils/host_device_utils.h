#pragma once

#include <core/macros.h>
#include <core/math/vec.h>

#include <cmath>

namespace pbal {

template <typename T>
FLUID_CUDA_HOST_DEVICE inline T area2(
    const Vec2<T>& p,
    const Vec2<T>& q,
    const Vec2<T>& s)
{

    return p.x * q.y - p.y * q.x
           + q.x * s.y - q.y * s.x
           + s.x * p.y - s.y * p.x;
}

template <typename T>
FLUID_CUDA_HOST_DEVICE bool toLeft(
    const Vec2<T>& p,
    const Vec2<T>& q,
    const Vec2<T>& s)
{

    return area2(p, q, s) > T();
}

template <typename T>
FLUID_CUDA_HOST_DEVICE inline T clamp(T val, T low, T high)
{
    if (val < low)
    {
        return low;
    }
    else if (val > high)
    {
        return high;
    }
    else
    {
        return val;
    }
}

template <typename S, typename T>
FLUID_CUDA_HOST_DEVICE inline S lerp(
    const S& v0,
    const S& v1,
    T        f)
{
    return (1 - f) * v0 + f * v1;
}

template <typename S, typename T>
FLUID_CUDA_HOST_DEVICE inline S bilerp(
    const S& f00,
    const S& f10,
    const S& f01,
    const S& f11,
    T        tx,
    T        ty)
{
    return lerp(
        lerp(f00, f10, tx),
        lerp(f01, f11, tx),
        ty);
}

template <typename S, typename T>
FLUID_CUDA_HOST_DEVICE inline S trilerp(
    const S& f000,
    const S& f100,
    const S& f010,
    const S& f110,
    const S& f001,
    const S& f101,
    const S& f011,
    const S& f111,
    T        tx,
    T        ty,
    T        fz)
{
    return lerp(
        bilerp(f000, f100, f010, f110, tx, ty),
        bilerp(f001, f101, f011, f111, tx, ty),
        fz);
}

template <typename T>
FLUID_CUDA_HOST_DEVICE inline void getBarycentric(
    T    x,
    int  iLow,
    int  iHigh,
    int* i,
    T*   f)
{

    T s = std::floor(x);
    *i  = static_cast<int>(s);

    int offset = -iLow;
    iLow += offset;
    iHigh += offset;

    if (iLow == iHigh)
    {
        *i = iLow;
        *f = 0;
    }
    else if (*i < iLow)
    {
        *i = iLow;
        *f = 0;
    }
    else if (*i > iHigh - 1)
    {
        *i = iHigh - 1;
        *f = 1;
    }
    else
    {
        *f = static_cast<T>(x - s);
    }

    *i -= offset;
}

template <typename T>
FLUID_CUDA_HOST_DEVICE inline void getCoordinatesAndWeights(
    const Vec2<T>& x,
    const Size2&   size,
    const Vec2<T>& spacing,
    const Vec2<T>& origin,
    Vec2i*         indices,
    T*             weights)
{

    int     i, j;
    T       fx, fy;
    Vec2<T> pt0 = (x - origin) / spacing;

    int iSize = static_cast<int>(size.x);
    int jSize = static_cast<int>(size.y);

    getBarycentric(pt0.x, 0, iSize - 1, &i, &fx);
    getBarycentric(pt0.y, 0, jSize - 1, &j, &fy);

    int ip1 = MIN(i + 1, iSize - 1);
    int jp1 = MIN(j + 1, jSize - 1);

    indices[0] = Vec2i(i, j);
    indices[1] = Vec2i(ip1, j);
    indices[2] = Vec2i(i, jp1);
    indices[3] = Vec2i(ip1, jp1);

    weights[0] = (1 - fx) * (1 - fy);
    weights[1] = fx * (1 - fy);
    weights[2] = (1 - fx) * fy;
    weights[3] = fx * fy;
}

template <typename T>
FLUID_CUDA_HOST_DEVICE inline void getCoordinatesAndWeights(
    const Vec3<T>& x,
    const Size3&   size,
    const Vec3<T>& spacing,
    const Vec3<T>& origin,
    Vec3i*         indices,
    T*             weights)
{

    int     i, j, k;
    T       fx, fy, fz;
    Vec3<T> pt0 = (x - origin) / spacing;

    int iSize = static_cast<int>(size.x);
    int jSize = static_cast<int>(size.y);
    int kSize = static_cast<int>(size.z);

    getBarycentric(pt0.x, 0, iSize - 1, &i, &fx);
    getBarycentric(pt0.y, 0, jSize - 1, &j, &fy);
    getBarycentric(pt0.z, 0, kSize - 1, &k, &fz);

    int ip1 = MIN(i + 1, iSize - 1);
    int jp1 = MIN(j + 1, jSize - 1);
    int kp1 = MIN(k + 1, kSize - 1);

    indices[0] = Vec3i(i, j, k);
    indices[1] = Vec3i(ip1, j, k);
    indices[2] = Vec3i(i, jp1, k);
    indices[3] = Vec3i(ip1, jp1, k);
    indices[4] = Vec3i(i, j, kp1);
    indices[5] = Vec3i(ip1, j, kp1);
    indices[6] = Vec3i(i, jp1, kp1);
    indices[7] = Vec3i(ip1, jp1, kp1);

    weights[0] = (1 - fx) * (1 - fy) * (1 - fz);
    weights[1] = fx * (1 - fy) * (1 - fz);
    weights[2] = (1 - fx) * fy * (1 - fz);
    weights[3] = fx * fy * (1 - fz);
    weights[4] = (1 - fx) * (1 - fy) * fz;
    weights[5] = fx * (1 - fy) * fz;
    weights[6] = (1 - fx) * fy * fz;
    weights[7] = fx * fy * fz;
}

}  // namespace pbal
