#pragma once

#include "def.cuh"

#define     GLH_ZERO            REAL(0.0)
#define     GLH_EPSILON         REAL(10e-6)
#define		GLH_EPSILON_2		REAL(10e-12)
#define     is_equal2(a,b)     (((a < b + GLH_EPSILON) && (a > b - GLH_EPSILON)) ? true : false)


inline __host__ __device__ REAL3 make_REAL3(REAL s)
{
	return make_REAL3(s, s, s);
}

inline __host__ __device__ REAL3 make_REAL3(const REAL s[])
{
	return make_REAL3(s[0], s[1], s[2]);
}

inline __host__ __device__ REAL getI(const REAL3 &a, int i)
{
	if (i == 0)
		return a.x;
	else if (i == 1)
		return a.y;
	else
		return a.z;
}

inline __host__ __device__ REAL3 zero3f()
{
	return make_REAL3(0, 0, 0);
}

inline __host__ __device__ void fswap(REAL &a, REAL &b)
{
	REAL t = b;
	b = a;
	a = t;
}

inline  __host__ __device__ REAL fminf(REAL a, REAL b)
{
	return a < b ? a : b;
}

inline  __host__ __device__ REAL fmaxf(REAL a, REAL b)
{
	return a > b ? a : b;
}

inline __host__ __device__ REAL3 fminf(const REAL3 &a, const REAL3 &b)
{
	return make_REAL3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}

inline __host__ __device__ REAL3 fmaxf(const REAL3 &a, const REAL3 &b)
{
	return make_REAL3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}

inline __host__ __device__ REAL3 operator-(const REAL3 &a, const REAL3 &b)
{
    return make_REAL3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ REAL2 operator-(const REAL2 &a, const REAL2 &b)
{
    return make_REAL2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ void operator-=(REAL3 &a, const REAL3 &b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

inline __host__ __device__ REAL3 cross(const REAL3 &a, const REAL3 &b)
{ 
    return make_REAL3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

inline __host__ __device__ REAL dot(const REAL3 &a, const REAL3 &b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ REAL dot(const REAL2 &a, const REAL2 &b)
{ 
    return a.x * b.x + a.y * b.y;
}

inline __host__ __device__ REAL stp(const REAL3 &u, const REAL3 &v, const REAL3 &w)
{
	return dot(u,cross(v,w));
}

inline __host__ __device__ REAL3 operator+(const REAL3 &a, const REAL3 &b)
{
    return make_REAL3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ REAL2 operator+(const REAL2 &a, const REAL2 &b)
{
    return make_REAL2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ void operator+=(REAL3 &a, REAL3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

inline __host__ __device__ void operator*=(REAL3 &a, REAL3 b)
{
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
}

inline __host__ __device__ void operator*=(REAL2 &a, REAL b)
{
    a.x *= b; a.y *= b;
}

inline __host__ __device__ REAL3 operator*(const REAL3 &a, REAL b)
{
    return make_REAL3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ REAL2 operator*(const REAL2 &a, REAL b)
{
    return make_REAL2(a.x * b, a.y * b);
}

inline __host__ __device__ REAL2 operator*(REAL b, const REAL2 &a)
{
    return make_REAL2(a.x * b, a.y * b);
}

inline __host__ __device__ REAL3 operator*(REAL b, const REAL3 &a)
{
    return make_REAL3(b * a.x, b * a.y, b * a.z);
}

inline __host__ __device__ void operator*=(REAL3 &a, REAL b)
{
    a.x *= b; a.y *= b; a.z *= b;
}

inline __host__ __device__ REAL3 operator/(const REAL3 &a, REAL b)
{
    return make_REAL3(a.x / b, a.y / b, a.z / b);
}

inline __host__ __device__ void operator/=(REAL3 &a, REAL b)
{
    a.x /= b; a.y /= b; a.z /= b;
}

inline __host__ __device__ REAL3 operator-(const REAL3 &a)
{
    return make_REAL3(-a.x, -a.y, -a.z);
}

#ifdef USE_DOUBLE
#ifndef P100
inline __device__ REAL atomicAddD(REAL* address, REAL val)
{
	unsigned long long int* address_as_ull =  (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed; 
	
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val + __longlong_as_double(assumed)));
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
inline __device__ REAL atomicAddD(REAL* address, REAL val)
{
	return atomicAdd(address, val);
}
#endif


inline __host__ __device__ REAL norm2(const REAL3 &v)
{
	return dot(v, v);
}

inline __host__ __device__ REAL length(const REAL3 &v)
{
    return sqrt(dot(v, v));
}


inline __host__ __device__ REAL3 normalize(const REAL3 &v)
{
    REAL invLen = rsqrt(dot(v, v));
    return v * invLen;
}

inline __device__ __host__ REAL3 lerp(const REAL3 &a, const REAL3 &b, REAL t)
{
    return a + t*(b-a);
}

inline __device__ __host__ REAL clamp(REAL x, REAL a, REAL b)
{
    return fminf(fmaxf(x, a), b);
}

inline __device__ __host__ REAL distance (const REAL3 &x, const REAL3 &a, const REAL3 &b) {
    REAL3 e = b-a;
    REAL3 xp = e*dot(e, x-a)/dot(e,e);
    // return norm((x-a)-xp);
    return fmaxf(length((x-a)-xp), REAL(1e-3)*length(e));
}

inline __device__ __host__ REAL2 barycentric_weights (const REAL3 &x, const REAL3 &a, const REAL3 &b) {
    REAL3 e = b-a;
    REAL t = dot(e, x-a)/dot(e,e);
    return make_REAL2(1-t, t);
}

inline __device__ void atomicAdd3(REAL3 *address, const REAL3 &val)
{
	atomicAddD(&address->x, val.x);
	atomicAddD(&address->y, val.y);
	atomicAddD(&address->z, val.z);
}

//////////////////////////////////////////////////////////////



typedef enum {
	RS_Positive = 0,
	RS_Negative = 1,
	RS_Unknow = 2
} RealSign;

inline __host__ __device__
RealSign rs_sign(REAL3 &a, REAL3 &b, REAL3 &c, REAL3 &axis)
{
	REAL ret = dot(axis, cross(a-c, b-c));

	if (ret > 0)
		return RS_Positive;

	if (ret < 0)
		return RS_Negative;

	return RS_Unknow;
}

inline 
__host__ __device__
RealSign rs_sign(REAL3 &a, REAL3 &va, REAL3 &b, REAL3 &c, REAL3 &axis)
{
	REAL ret1 = dot(axis, cross(a-c, b-c));
	REAL ret2 = dot(axis, cross(a+va-c, b-c));

	if (ret1 > 0 && ret2 > 0)
		return RS_Positive;

	if (ret1 < 0 && ret2 < 0)
		return RS_Negative;

	return RS_Unknow;
}

inline __host__ __device__
RealSign rs_sign(REAL3 &a, REAL3 &b0, REAL3 &vb, REAL3 &c0, REAL3 &vc, REAL3 &axis)
{
	REAL A = dot(axis, cross(b0 - a, c0 - a));
	REAL B = dot(axis, cross(vb, c0 - a) + cross(b0 - a, vc));
	REAL C = dot(axis, cross(vb, vc));

	REAL U = A;
	REAL V = B*REAL(0.5) + A;
	REAL W = A + B + C;

	if (U > 0 && V > 0 && W > 0)
		return RS_Positive;

	if (U < 0 && V < 0 && W < 0)
		return RS_Negative;

	return RS_Unknow;
}
