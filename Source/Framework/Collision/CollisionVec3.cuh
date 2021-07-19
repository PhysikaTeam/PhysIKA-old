/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2021-05-30
 * @description: device vec3f data structure
 * @version    : 1.0
 */

#pragma once

#include <cuda_runtime.h>
#include "Core/Utility/cuda_helper_math.h"

#define GLH_ZERO float(0.0)
#define GLH_EPSILON float(10e-6)
#define GLH_EPSILON_2 float(10e-12)
#define is_equal2(a, b) (((a < b + GLH_EPSILON) && (a > b - GLH_EPSILON)) ? true : false)

/**
  * warpper of atomic add function to handle different precision in different gpu
  *
  * @param address address of the value to be add
  * @param val     increnmental value
  * @return old value
  */
inline __host__ __device__ float3 make_float3(const float s[])
{
    return make_float3(s[0], s[1], s[2]);
}

/**
 * get the value of the i-th dimension of the vector-3d
 *
 * @param a  vector-3d
 * @param i  index
 * @return the value of i-th dimension
 */
inline __host__ __device__ float getI(const float3& a, int i)
{
    if (i == 0)
        return a.x;
    else if (i == 1)
        return a.y;
    else
        return a.z;
}

/**
 * get a vector-3d with all elements set to be zero
 *
 * @return zero vector-3d
 */
inline __host__ __device__ float3 zero3f()
{
    return make_float3(0, 0, 0);
}

/**
 * swap two vector
 *
 * @param[in, out] a vector to be swap
 * @param[in, out] b vector to be swap
 */
inline __host__ __device__ void fswap(float& a, float& b)
{
    float t = b;
    b       = a;
    a       = t;
}

/**
 * get the volumn of the hexahedron defined by u, v, w
 *
 * @param[in] u one aixs of the hexahedron
 * @param[in] v one aixs of the hexahedron
 * @param[in] w one aixs of the hexahedron
 * @return the volumn of the hexahedron
 */
inline __host__ __device__ float stp(const float3& u, const float3& v, const float3& w)
{
    return dot(u, cross(v, w));
}

/**
 * get the negated vector-3d 
 *
 * @param[in] a original vector-3d
 * @return the negated result
 */
inline __host__ __device__ float3 operator-(const float3& a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

/**
 * warpper of atomic add function to handle different precision in different gpu
 * 
 * @param address address of the value to be add
 * @param val     increnmental value
 * @return old value
 */
#ifdef USE_double
#ifndef P100
inline __device__ REAL atomicAddD(REAL* address, REAL val)
{
    unsigned long long int* address_as_ull = ( unsigned long long int* )address;
    unsigned long long int  old            = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old     = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}
#else

inline __device__ REAL atomicAddD(REAL* address, REAL val)
{
    return atomicAdd(address, val);
}

#endif
#else
inline __device__ float atomicAddD(float* address, float val)
{
    return atomicAdd(address, val);
}
#endif

/**
 * square length of the vector-3d
 *
 * @param address address of the value to be add
 * @param val     increnmental value
 * @return old value
 */
inline __host__ __device__ float norm2(const float3& v)
{
    return dot(v, v);
}

/**
 * square length of the vector-3d
 *
 * @param address address of the value to be add
 * @param val     increnmental value
 * @return old value
 */
inline __device__ __host__ float3 lerp(const float3& a, const float3& b, float t)
{
    return a + t * (b - a);
}

/**
 * square length of the vector-3d
 *
 * @param address address of the value to be add
 * @param val     increnmental value
 * @return old value
 */
inline __device__ __host__ float distance(const float3& x, const float3& a, const float3& b)
{
    float3 e  = b - a;
    float3 xp = e * dot(e, x - a) / dot(e, e);
    // return norm((x-a)-xp);
    return fmaxf(length((x - a) - xp), float(1e-3) * length(e));
}

/**
 * square length of the vector-3d
 *
 * @param address address of the value to be add
 * @param val     increnmental value
 * @return old value
 */
inline __device__ __host__ float2 barycentric_weights(const float3& x, const float3& a, const float3& b)
{
    float3 e = b - a;
    float  t = dot(e, x - a) / dot(e, e);
    return make_float2(1 - t, t);
}

/**
 * square length of the vector-3d
 *
 * @param address address of the value to be add
 * @param val     increnmental value
 * @return old value
 */
inline __device__ void atomicAdd3(float3* address, const float3& val)
{
    atomicAddD(&address->x, val.x);
    atomicAddD(&address->y, val.y);
    atomicAddD(&address->z, val.z);
}