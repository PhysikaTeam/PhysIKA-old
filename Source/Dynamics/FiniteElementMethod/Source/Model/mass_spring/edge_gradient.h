/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: edge_gradient for mass spring method.
 * @version    : 1.0
 */
#ifndef EDGE_GRADIENT_JJ_H
#define EDGE_GRADIENT_JJ_H

/* edge_gradient.f -- translated by f2c (version 20160102).
   You must link the resulting object file with libf2c:
    on Microsoft Windows system, link with libf2c.lib;
    on Linux or Unix systems, link with .../path/to/libf2c.a -lm
    or, if you install libf2c.a in a standard place, with -lf2c -lm
    -- in that order, at the end of the command line, as in
        cc *.o -lf2c -lm
    Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

        http://www.netlib.org/f2c/libf2c.zip
*/

// #include "f2c.h"
#include <cmath>

template <typename T>
/* Subroutine */ int EdgeGradient(T* __restrict x, T* __restrict k, T* __restrict l0, T* __restrict g)
{
    /* System generated locals */
    T r__1, r__2, r__3, r__4, r__5, r__6;

    /* Builtin functions */
    // double sqrt(doublereal);

    /* Parameter adjustments */
    --g;
    --x;

    /* Function Body */
    /* Computing 2nd power */
    r__1 = x[6] - x[3];
    /* Computing 2nd power */
    r__2 = x[5] - x[2];
    /* Computing 2nd power */
    r__3 = x[4] - x[1];
    /* Computing 2nd power */
    r__4 = x[6] - x[3];
    /* Computing 2nd power */
    r__5 = x[5] - x[2];
    /* Computing 2nd power */
    r__6 = x[4] - x[1];
    g[1] = -((x[4] - x[1]) * *k * (sqrt(r__1 * r__1 + r__2 * r__2 + r__3 * r__3) - *l0)) / (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) * *l0);
    /* Computing 2nd power */
    r__1 = x[6] - x[3];
    /* Computing 2nd power */
    r__2 = x[5] - x[2];
    /* Computing 2nd power */
    r__3 = x[4] - x[1];
    /* Computing 2nd power */
    r__4 = x[6] - x[3];
    /* Computing 2nd power */
    r__5 = x[5] - x[2];
    /* Computing 2nd power */
    r__6 = x[4] - x[1];
    g[2] = -((x[5] - x[2]) * *k * (sqrt(r__1 * r__1 + r__2 * r__2 + r__3 * r__3) - *l0)) / (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) * *l0);
    /* Computing 2nd power */
    r__1 = x[6] - x[3];
    /* Computing 2nd power */
    r__2 = x[5] - x[2];
    /* Computing 2nd power */
    r__3 = x[4] - x[1];
    /* Computing 2nd power */
    r__4 = x[6] - x[3];
    /* Computing 2nd power */
    r__5 = x[5] - x[2];
    /* Computing 2nd power */
    r__6 = x[4] - x[1];
    g[3] = -((x[6] - x[3]) * *k * (sqrt(r__1 * r__1 + r__2 * r__2 + r__3 * r__3) - *l0)) / (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) * *l0);
    /* Computing 2nd power */
    r__1 = x[6] - x[3];
    /* Computing 2nd power */
    r__2 = x[5] - x[2];
    /* Computing 2nd power */
    r__3 = x[4] - x[1];
    /* Computing 2nd power */
    r__4 = x[6] - x[3];
    /* Computing 2nd power */
    r__5 = x[5] - x[2];
    /* Computing 2nd power */
    r__6 = x[4] - x[1];
    g[4] = (x[4] - x[1]) * *k * (sqrt(r__1 * r__1 + r__2 * r__2 + r__3 * r__3) - *l0) / (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) * *l0);
    /* Computing 2nd power */
    r__1 = x[6] - x[3];
    /* Computing 2nd power */
    r__2 = x[5] - x[2];
    /* Computing 2nd power */
    r__3 = x[4] - x[1];
    /* Computing 2nd power */
    r__4 = x[6] - x[3];
    /* Computing 2nd power */
    r__5 = x[5] - x[2];
    /* Computing 2nd power */
    r__6 = x[4] - x[1];
    g[5] = (x[5] - x[2]) * *k * (sqrt(r__1 * r__1 + r__2 * r__2 + r__3 * r__3) - *l0) / (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) * *l0);
    /* Computing 2nd power */
    r__1 = x[6] - x[3];
    /* Computing 2nd power */
    r__2 = x[5] - x[2];
    /* Computing 2nd power */
    r__3 = x[4] - x[1];
    /* Computing 2nd power */
    r__4 = x[6] - x[3];
    /* Computing 2nd power */
    r__5 = x[5] - x[2];
    /* Computing 2nd power */
    r__6 = x[4] - x[1];
    g[6] = (x[6] - x[3]) * *k * (sqrt(r__1 * r__1 + r__2 * r__2 + r__3 * r__3) - *l0) / (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) * *l0);
    return 0;
} /* edge_gradient__ */

template int EdgeGradient<double>(double* __restrict x, double* __restrict l0, double* __restrict k, double* __restrict g);
template int EdgeGradient<float>(float* __restrict x, float* __restrict l0, float* __restrict k, float* __restrict g);

#endif  // EDGE_GRADIENT_JJ_H
