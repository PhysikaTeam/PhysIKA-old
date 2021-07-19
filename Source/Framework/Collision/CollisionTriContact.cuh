/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2021-05-30
 * @description: very robust triangle intersection test uses no divisions and 
 *               works on coplanar triangles, this is an internal header and 
 *               should not be used directly
 * @version    : 1.0
 */

#pragma once

#include "CollisionTools.cuh"
#include "CollisionVec3.cuh"

inline __device__ int project3(const float3& ax,
                               const float3& p1,
                               const float3& p2,
                               const float3& p3)
{
    float P1 = dot(ax, p1);
    float P2 = dot(ax, p2);
    float P3 = dot(ax, p3);

    float mx1 = fmaxf(P1, fmaxf(P2, P3));
    float mn1 = fminf(P1, fminf(P2, P3));

    if (mn1 > 0)
        return 0;
    if (0 > mx1)
        return 0;

    return 1;
}

inline __device__ int project6(float3& ax,
                               float3& p1,
                               float3& p2,
                               float3& p3,
                               float3& q1,
                               float3& q2,
                               float3& q3)
{
    float P1 = dot(ax, p1);
    float P2 = dot(ax, p2);
    float P3 = dot(ax, p3);
    float Q1 = dot(ax, q1);
    float Q2 = dot(ax, q2);
    float Q3 = dot(ax, q3);

    float mx1 = fmaxf(P1, fmaxf(P2, P3));
    float mn1 = fminf(P1, fminf(P2, P3));
    float mx2 = fmaxf(Q1, fmaxf(Q2, Q3));
    float mn2 = fminf(Q1, fminf(Q2, Q3));

    if (mn1 > mx2)
        return 0;
    if (mn2 > mx1)
        return 0;

    return 1;
}

inline __device__ bool
tri_contact(float3& P1, float3& P2, float3& P3, float3& Q1, float3& Q2, float3& Q3)
{
    float3 p1 = zero3f();
    ;
    float3 p2 = P2 - P1;
    float3 p3 = P3 - P1;
    float3 q1 = Q1 - P1;
    float3 q2 = Q2 - P1;
    float3 q3 = Q3 - P1;

    float3 e1 = p2 - p1;
    float3 e2 = p3 - p2;
    float3 e3 = p1 - p3;

    float3 f1 = q2 - q1;
    float3 f2 = q3 - q2;
    float3 f3 = q1 - q3;

    float3 n1 = cross(e1, e2);
    float3 m1 = cross(f1, f2);

    float3 g1 = cross(e1, n1);
    float3 g2 = cross(e2, n1);
    float3 g3 = cross(e3, n1);

    float3 h1 = cross(f1, m1);
    float3 h2 = cross(f2, m1);
    float3 h3 = cross(f3, m1);

    float3 ef11 = cross(e1, f1);
    float3 ef12 = cross(e1, f2);
    float3 ef13 = cross(e1, f3);
    float3 ef21 = cross(e2, f1);
    float3 ef22 = cross(e2, f2);
    float3 ef23 = cross(e2, f3);
    float3 ef31 = cross(e3, f1);
    float3 ef32 = cross(e3, f2);
    float3 ef33 = cross(e3, f3);

    // now begin the series of tests
    if (!project3(n1, q1, q2, q3))
        return false;
    if (!project3(m1, -q1, p2 - q1, p3 - q1))
        return false;

    if (!project6(ef11, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef12, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef13, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef21, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef22, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef23, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef31, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef32, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef33, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(g1, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(g2, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(g3, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(h1, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(h2, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(h3, p1, p2, p3, q1, q2, q3))
        return false;

    return true;
}