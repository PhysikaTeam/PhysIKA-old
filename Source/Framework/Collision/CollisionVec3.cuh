#pragma once

#include <cuda_runtime.h>
#include "Core/Utility/cuda_helper_math.h"

#define     GLH_ZERO            float(0.0)
#define     GLH_EPSILON         float(10e-6)
#define		GLH_EPSILON_2		float(10e-12)
#define     is_equal2(a,b)     (((a < b + GLH_EPSILON) && (a > b - GLH_EPSILON)) ? true : false)


//inline __host__ __device__ float3 make_float3(float s)
//{
//	return make_float3(s, s, s);
//}

inline __host__ __device__ float3 make_float3(const float s[])
{
	return make_float3(s[0], s[1], s[2]);
}

inline __host__ __device__ float getI(const float3& a, int i)
{
	if (i == 0)
		return a.x;
	else if (i == 1)
		return a.y;
	else
		return a.z;
}

inline __host__ __device__ float3 zero3f()
{
	return make_float3(0, 0, 0);
}

inline __host__ __device__ void fswap(float& a, float& b)
{
	float t = b;
	b = a;
	a = t;
}

//inline  __host__ __device__ float fminf(float a, float b)
//{
//	return a < b ? a : b;
//}
//
//inline  __host__ __device__ float fmaxf(float a, float b)
//{
//	return a > b ? a : b;
//}

//inline __host__ __device__ float3 fminf(const float3& a, const float3& b)
//{
//	return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
//}
//
//inline __host__ __device__ float3 fmaxf(const float3& a, const float3& b)
//{
//	return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
//}

//inline __host__ __device__ float3 operator-(const float3& a, const float3& b)
//{
//	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
//}
//
//inline __host__ __device__ float2 operator-(const float2& a, const float2& b)
//{
//	return make_float2(a.x - b.x, a.y - b.y);
//}
//
//inline __host__ __device__ void operator-=(float3& a, const float3& b)
//{
//	a.x -= b.x; a.y -= b.y; a.z -= b.z;
//}

//inline __host__ __device__ float3 cross(const float3& a, const float3& b)
//{
//	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
//}
//
//inline __host__ __device__ float dot(const float3& a, const float3& b)
//{
//	return a.x * b.x + a.y * b.y + a.z * b.z;
//}

//inline __host__ __device__ float dot(const float2& a, const float2& b)
//{
//	return a.x * b.x + a.y * b.y;
//}

inline __host__ __device__ float stp(const float3& u, const float3& v, const float3& w)
{
	return dot(u, cross(v, w));
}

//inline __host__ __device__ float3 operator+(const float3& a, const float3& b)
//{
//	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
//}

//inline __host__ __device__ float2 operator+(const float2& a, const float2& b)
//{
//	return make_float2(a.x + b.x, a.y + b.y);
//}

//inline __host__ __device__ void operator+=(float3& a, float3 b)
//{
//	a.x += b.x; a.y += b.y; a.z += b.z;
//}

//inline __host__ __device__ void operator*=(float3& a, float3 b)
//{
//	a.x *= b.x; a.y *= b.y; a.z *= b.z;
//}

//inline __host__ __device__ void operator*=(float2& a, float b)
//{
//	a.x *= b; a.y *= b;
//}

//inline __host__ __device__ float3 operator*(const float3& a, float b)
//{
//	return make_float3(a.x * b, a.y * b, a.z * b);
//}

//inline __host__ __device__ float2 operator*(const float2& a, float b)
//{
//	return make_float2(a.x * b, a.y * b);
//}
//
//inline __host__ __device__ float2 operator*(float b, const float2& a)
//{
//	return make_float2(a.x * b, a.y * b);
//}
//
//inline __host__ __device__ float3 operator*(float b, const float3& a)
//{
//	return make_float3(b * a.x, b * a.y, b * a.z);
//}

//inline __host__ __device__ void operator*=(float3& a, float b)
//{
//	a.x *= b; a.y *= b; a.z *= b;
//}

//inline __host__ __device__ float3 operator/(const float3& a, float b)
//{
//	return make_float3(a.x / b, a.y / b, a.z / b);
//}

//inline __host__ __device__ void operator/=(float3& a, float b)
//{
//	a.x /= b; a.y /= b; a.z /= b;
//}

inline __host__ __device__ float3 operator-(const float3& a)
{
	return make_float3(-a.x, -a.y, -a.z);
}

#ifdef USE_double
#ifndef P100
inline __device__ REAL atomicAddD(REAL* address, REAL val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
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


inline __host__ __device__ float norm2(const float3& v)
{
	return dot(v, v);
}

//inline __host__ __device__ float length(const float3& v)
//{
//	return sqrt(dot(v, v));
//}


//inline __host__ __device__ float3 normalize(const float3& v)
//{
//	float invLen = rsqrt(dot(v, v));
//	return v * invLen;
//}

inline __device__ __host__ float3 lerp(const float3& a, const float3& b, float t)
{
	return a + t * (b - a);
}

//inline __device__ __host__ float clamp(float x, float a, float b)
//{
//	return fminf(fmaxf(x, a), b);
//}

inline __device__ __host__ float distance(const float3& x, const float3& a, const float3& b) {
	float3 e = b - a;
	float3 xp = e * dot(e, x - a) / dot(e, e);
	// return norm((x-a)-xp);
	return fmaxf(length((x - a) - xp), float(1e-3) * length(e));
}

inline __device__ __host__ float2 barycentric_weights(const float3& x, const float3& a, const float3& b) {
	float3 e = b - a;
	float t = dot(e, x - a) / dot(e, e);
	return make_float2(1 - t, t);
}

inline __device__ void atomicAdd3(float3* address, const float3& val)
{
	atomicAddD(&address->x, val.x);
	atomicAddD(&address->y, val.y);
	atomicAddD(&address->z, val.z);
}