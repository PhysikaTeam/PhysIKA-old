/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2021-05-30
 * @description: some device data structures for
 * @version    : 1.0
 */

#pragma once

#include "CollisionVec3.cuh"
#include "CollisionTools.cuh"

#define EPS_N float(0.000001)
#define HALF_PI float(0.5 * M_PI)

/**
  * device cone data structure, use axis and angle to determine the shape
  */
typedef struct __align__(16) _cone3f
{
    float3 _axis;   //!< axis of the cone
    float  _angle;  //!< angle of the cone

    /**
	 * check if the cone is empty
	 *
	 * @return whether the cone is empty
	 */
    inline __host__ __device__ void empty()
    {
        _axis  = zero3f();
        _angle = 0.0;
    }

    /**
	 * set the axis of the cone and initialize the angle
	 *
	 * @param[in] v the axis of the cone
	 */
    inline __host__ __device__ void set(const float3& v)
    {
        _axis  = v;
        _angle = EPS_N;
    }

    /**
	 * copy constructor
	 *
	 * @param[in] n the other cone
	 */
    inline __host__ __device__ void set(const _cone3f& n)
    {
        _axis  = n._axis;
        _angle = n._angle;
    }

    /**
	 * set the axis of the cone and initialize the angle
	 *
	 * @param[in] v the axis of the cone
	 */
    inline __host__ __device__ void set(const float3& v1, const float3& v2)
    {
        _axis  = normalize(v1 + v2);
        _angle = acos(dot(v1, v2)) * float(0.5);
    }

    inline __host__ __device__ bool full() const
    {
        return _angle >= HALF_PI;
    }

    inline __host__ __device__ void set_full()
    {
        _angle = HALF_PI + EPS_N;
    }
}
g_cone;

inline __host__ __device__ void operator+=(g_cone& a, const float3& v)
{
    if (a.full())
        return;

    float vdot = dot(v, a._axis);

    if (vdot < 0)
    {
        a.set_full();
        return;
    }

    float angle = acos(vdot);
    if (angle <= a._angle)
        return;

    a._axis = normalize(a._axis + v);
    a._angle += angle * float(0.5);
}

inline __host__ __device__ void operator+=(g_cone& a, g_cone& b)
{
    if (a.full())
        return;

    if (b.full())
    {
        a.set_full();
        return;
    }

    float vdot = dot(a._axis, b._axis);
    if (vdot < 0)
    {
        a.set_full();
        return;
    }

    float angle      = acos(vdot);
    float diff_angle = fabs(a._angle - b._angle);
    if (angle <= diff_angle)
    {
        if (b._angle > a._angle)
            a.set(b);

        return;
    }

    a._axis  = normalize(a._axis + b._axis);
    a._angle = angle * float(0.5) + fmaxf(a._angle, b._angle);
}

/**
 * device aabb box using float3 as internal storage
 */
typedef struct __align__(16) _box3f
{
    float3 _min, _max;

    /**
	 * set the min and max boundary with the input 3d-vector
	 *
	 * @param[in] a the min and max boundary of the aabb box
	 */
    inline __host__ __device__ void set(const float3& a)
    {
        _min = _max = a;
    }

    /**
	 * set the min and max boundary with the two input 3d-vectors
	 *
	 * @param[in] a one 3d-vector to determine boundary of the aabb box
	 * @param[in] b the other 3d-vector to determine boundary of the aabb box
	 */
    inline __host__ __device__ void set(const float3& a, const float3& b)
    {
        _min = fminf(a, b);
        _max = fmaxf(a, b);
    }

    /**
	 * set the min and max boundary with the input boxes
	 *
	 * @param[in] a one of the aabb box
	 * @param[in] b the other aabb box
	 */
    inline __host__ __device__ void set(const _box3f& a, const _box3f& b)
    {
        _min = fminf(a._min, b._min);
        _max = fmaxf(a._max, b._max);
    }

    /**
	 * enlarge the box to contain a point
	 *
	 * @param[in] a the position of the point
	 */
    inline __host__ __device__ void add(const float3& a)
    {
        _min = fminf(_min, a);
        _max = fmaxf(_max, a);
    }

    /**
	 * set the min and max boundary to the input 3d-vector
	 *
	 * @param[in] a the min and max boundary of the aabb box
	 */
    inline __host__ __device__ void add(const _box3f& b)
    {
        _min = fminf(_min, b._min);
        _max = fmaxf(_max, b._max);
    }

    /**
	 * enlarge the aabb box with a thickness in all directions
	 *
	 * @param[in] thickness the incremental value to enlarge the box
	 */
    inline __host__ __device__ void enlarge(float thickness)
    {
        _min -= make_float3(thickness);
        _max += make_float3(thickness);
    }

    /**
	 * check if the current box overlaps the given box
	 *
	 * @param[in] b the given box
	 *
	 * @return whether two boxes are overlaped
	 */
    inline __host__ __device__ bool overlaps(const _box3f& b) const
    {
        if (_min.x > b._max.x)
            return false;
        if (_min.y > b._max.y)
            return false;
        if (_min.z > b._max.z)
            return false;

        if (_max.x < b._min.x)
            return false;
        if (_max.y < b._min.y)
            return false;
        if (_max.z < b._min.z)
            return false;

        return true;
    }

    /**
	 * get max boundary of the aabb box
	 *
	 * @return float3 of max boundary of box
	 */
    inline __host__ __device__
        float3
        maxV() const
    {
        return _max;
    }

    /**
	 * get max boundary of the aabb box
	 *
	 * @return float3 of max boundary of box
	 */
    inline __host__ __device__
        float3
        minV() const
    {
        return _min;
    }

    /**
	 * print the min and max boundary of the aabb box to console
	 */
    inline void print()
    {
        printf("%f, %f, %f, %f, %f, %f\n", _min.x, _min.y, _min.z, _max.x, _max.y, _max.z);
    }

    /**
	 * print the min and max boundary of the aabb box to file
	 */
    inline void print(FILE * fp)
    {
        fprintf(fp, "%f, %f, %f, %f, %f, %f\n", _min.x, _min.y, _min.z, _max.x, _max.y, _max.z);
    }
}
g_box;