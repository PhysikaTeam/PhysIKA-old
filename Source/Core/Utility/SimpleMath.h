#ifndef SIMPLE_MATH_H
#define SIMPLE_MATH_H

#include <cuda_runtime.h>

#include "Core/Vector.h"
#include "Core/Matrix.h"

namespace PhysIKA {
template <typename Scalar>
inline __host__ __device__ Scalar clamp(const Scalar& v, const Scalar& lo, const Scalar& hi)
{
    return (v < lo) ? lo : (hi < v) ? hi
                                    : v;
}

template <typename Scalar>
inline __host__ __device__ Vector<Scalar, 2> clamp(const Vector<Scalar, 2>& v, const Vector<Scalar, 2>& lo, const Vector<Scalar, 2>& hi)
{
    Vector<Scalar, 2> ret;
    ret[0] = (v[0] < lo[0]) ? lo[0] : ((hi[0] < v[0]) ? hi[0] : v[0]);
    ret[1] = (v[1] < lo[1]) ? lo[1] : ((hi[1] < v[1]) ? hi[1] : v[1]);

    return ret;
}

template <typename Scalar>
inline __host__ __device__ Vector<Scalar, 3> clamp(const Vector<Scalar, 3>& v, const Vector<Scalar, 3>& lo, const Vector<Scalar, 3>& hi)
{
    Vector<Scalar, 3> ret;
    ret[0] = (v[0] < lo[0]) ? lo[0] : ((hi[0] < v[0]) ? hi[0] : v[0]);
    ret[1] = (v[1] < lo[1]) ? lo[1] : ((hi[1] < v[1]) ? hi[1] : v[1]);
    ret[2] = (v[2] < lo[2]) ? lo[2] : ((hi[2] < v[2]) ? hi[2] : v[2]);

    return ret;
}

template <typename Scalar>
inline __host__ __device__ Vector<Scalar, 4> clamp(const Vector<Scalar, 4>& v, const Vector<Scalar, 4>& lo, const Vector<Scalar, 4>& hi)
{
    Vector<Scalar, 3> ret;
    ret[0] = (v[0] < lo[0]) ? lo[0] : ((hi[0] < v[0]) ? hi[0] : v[0]);
    ret[1] = (v[1] < lo[1]) ? lo[1] : ((hi[1] < v[1]) ? hi[1] : v[1]);
    ret[2] = (v[2] < lo[2]) ? lo[2] : ((hi[2] < v[2]) ? hi[2] : v[2]);
    ret[3] = (v[3] < lo[3]) ? lo[3] : ((hi[3] < v[3]) ? hi[3] : v[3]);

    return ret;
}

template <typename Scalar>
inline __host__ __device__ Scalar abs(const Scalar& v)
{
    return v < Scalar(0) ? -v : v;
}

template <typename Scalar>
inline __host__ __device__ Vector<Scalar, 2> abs(const Vector<Scalar, 2>& v)
{
    Vector<Scalar, 2> ret;
    ret[0] = (v[0] < Scalar(0)) ? -v[0] : v[0];
    ret[1] = (v[1] < Scalar(0)) ? -v[1] : v[1];

    return ret;
}

template <typename Scalar>
inline __host__ __device__ Vector<Scalar, 3> abs(const Vector<Scalar, 3>& v)
{
    Vector<Scalar, 3> ret;
    ret[0] = (v[0] < Scalar(0)) ? -v[0] : v[0];
    ret[1] = (v[1] < Scalar(0)) ? -v[1] : v[1];
    ret[2] = (v[2] < Scalar(0)) ? -v[2] : v[2];

    return ret;
}

template <typename Scalar>
inline __host__ __device__ Vector<Scalar, 4> abs(const Vector<Scalar, 4>& v)
{
    Vector<Scalar, 3> ret;
    ret[0] = (v[0] < Scalar(0)) ? -v[0] : v[0];
    ret[1] = (v[1] < Scalar(0)) ? -v[1] : v[1];
    ret[2] = (v[2] < Scalar(0)) ? -v[2] : v[2];
    ret[3] = (v[3] < Scalar(0)) ? -v[3] : v[3];

    return ret;
}

template <typename Scalar>
inline __host__ __device__ Scalar minimum(const Scalar& v0, const Scalar& v1)
{
    return v0 < v1 ? v0 : v1;
}

template <typename Scalar>
inline __host__ __device__ Vector<Scalar, 2> minimum(const Vector<Scalar, 2>& v0, const Vector<Scalar, 2>& v1)
{
    Vector<Scalar, 2> ret;
    ret[0] = (v0[0] < v1[0]) ? v0[0] : v1[0];
    ret[1] = (v0[1] < v1[1]) ? v0[1] : v1[1];

    return ret;
}

template <typename Scalar>
inline __host__ __device__ Vector<Scalar, 3> minimum(const Vector<Scalar, 3>& v0, const Vector<Scalar, 3>& v1)
{
    Vector<Scalar, 3> ret;
    ret[0] = (v0[0] < v1[0]) ? v0[0] : v1[0];
    ret[1] = (v0[1] < v1[1]) ? v0[1] : v1[1];
    ret[2] = (v0[2] < v1[2]) ? v0[2] : v1[2];

    return ret;
}

template <typename Scalar>
inline __host__ __device__ Vector<Scalar, 4> minimum(const Vector<Scalar, 4>& v0, const Vector<Scalar, 4>& v1)
{
    Vector<Scalar, 4> ret;
    ret[0] = (v0[0] < v1[0]) ? v0[0] : v1[0];
    ret[1] = (v0[1] < v1[1]) ? v0[1] : v1[1];
    ret[2] = (v0[2] < v1[2]) ? v0[2] : v1[2];
    ret[3] = (v0[3] < v1[3]) ? v0[3] : v1[3];

    return ret;
}

template <typename Scalar>
inline __host__ __device__ Scalar maximum(const Scalar& v0, const Scalar& v1)
{
    return v0 > v1 ? v0 : v1;
}

template <typename Scalar>
inline __host__ __device__ Vector<Scalar, 2> maximum(const Vector<Scalar, 2>& v0, const Vector<Scalar, 2>& v1)
{
    Vector<Scalar, 2> ret;
    ret[0] = (v0[0] > v1[0]) ? v0[0] : v1[0];
    ret[1] = (v0[1] > v1[1]) ? v0[1] : v1[1];

    return ret;
}

template <typename Scalar>
inline __host__ __device__ Vector<Scalar, 3> maximum(const Vector<Scalar, 3>& v0, const Vector<Scalar, 3>& v1)
{
    Vector<Scalar, 3> ret;
    ret[0] = (v0[0] > v1[0]) ? v0[0] : v1[0];
    ret[1] = (v0[1] > v1[1]) ? v0[1] : v1[1];
    ret[2] = (v0[2] > v1[2]) ? v0[2] : v1[2];

    return ret;
}

template <typename Scalar>
inline __host__ __device__ Vector<Scalar, 4> maximum(const Vector<Scalar, 4>& v0, const Vector<Scalar, 4>& v1)
{
    Vector<Scalar, 4> ret;
    ret[0] = (v0[0] > v1[0]) ? v0[0] : v1[0];
    ret[1] = (v0[1] > v1[1]) ? v0[1] : v1[1];
    ret[2] = (v0[2] > v1[2]) ? v0[2] : v1[2];
    ret[3] = (v0[3] > v1[3]) ? v0[3] : v1[3];

    return ret;
}
}  // namespace PhysIKA

#endif