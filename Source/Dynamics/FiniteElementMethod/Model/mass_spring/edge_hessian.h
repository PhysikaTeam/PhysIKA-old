/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: edge hessian for mass spring method.
 * @version    : 1.0
 */
#ifndef EDGE_HESSIAN_JJ_H
#define EDGE_HESSIAN_JJ_H

/* edge_hessian.f -- translated by f2c (version 20160102).
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

/* Table of constant values */

#include <cmath>

template <typename T>
T pow_dd(T* __restrict b, T* __restrict a)
{
    return pow(*b, *a);
}

template <typename T>
/* Subroutine */ int EdgeHessian(T* __restrict x, T* __restrict k, T* __restrict l0, T* __restrict h__)
{

    T c_b2 = -1.5;

    /* System generated locals */
    T r__1, r__2, r__3, r__4, r__5, r__6, r__7, r__8, r__9, r__10, r__11,
        r__12, r__13, r__14, r__15, r__16, r__17;
    T d__1;

    /* Builtin functions */

    /* Parameter adjustments */
    h__ -= 7;
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
    /* Computing 2nd power */
    r__7 = x[4] - x[1];
    /* Computing 2nd power */
    r__8 = x[6] - x[3];
    /* Computing 2nd power */
    r__9 = x[5] - x[2];
    /* Computing 2nd power */
    r__10 = x[4] - x[1];
    d__1  = (T)(r__8 * r__8 + r__9 * r__9 + r__10 * r__10);
    /* Computing 2nd power */
    r__11 = x[6] - x[3];
    /* Computing 2nd power */
    r__12 = x[5] - x[2];
    /* Computing 2nd power */
    r__13 = x[4] - x[1];
    /* Computing 2nd power */
    r__14 = x[4] - x[1];
    /* Computing 2nd power */
    r__15 = x[6] - x[3];
    /* Computing 2nd power */
    r__16 = x[5] - x[2];
    /* Computing 2nd power */
    r__17  = x[4] - x[1];
    h__[7] = *k * (sqrt(r__1 * r__1 + r__2 * r__2 + r__3 * r__3) - *l0) / (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) * *l0) - r__7 * r__7 * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__11 * r__11 + r__12 * r__12 + r__13 * r__13) - *l0) / *l0 + r__14 * r__14 * *k / ((r__15 * r__15 + r__16 * r__16 + r__17 * r__17) * *l0);
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
    d__1 = (T)(r__4 * r__4 + r__5 * r__5 + r__6 * r__6);
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[13] = (x[4] - x[1]) * (x[5] - x[2]) * *k / ((r__1 * r__1 + r__2 * r__2 + r__3 * r__3) * *l0) - (x[4] - x[1]) * (x[5] - x[2]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__7 * r__7 + r__8 * r__8 + r__9 * r__9) - *l0) / *l0;
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
    d__1 = (T)(r__4 * r__4 + r__5 * r__5 + r__6 * r__6);
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[19] = (x[4] - x[1]) * (x[6] - x[3]) * *k / ((r__1 * r__1 + r__2 * r__2 + r__3 * r__3) * *l0) - (x[4] - x[1]) * (x[6] - x[3]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__7 * r__7 + r__8 * r__8 + r__9 * r__9) - *l0) / *l0;
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
    /* Computing 2nd power */
    r__7 = x[4] - x[1];
    /* Computing 2nd power */
    r__8 = x[6] - x[3];
    /* Computing 2nd power */
    r__9 = x[5] - x[2];
    /* Computing 2nd power */
    r__10 = x[4] - x[1];
    d__1  = (T)(r__8 * r__8 + r__9 * r__9 + r__10 * r__10);
    /* Computing 2nd power */
    r__11 = x[6] - x[3];
    /* Computing 2nd power */
    r__12 = x[5] - x[2];
    /* Computing 2nd power */
    r__13 = x[4] - x[1];
    /* Computing 2nd power */
    r__14 = x[4] - x[1];
    /* Computing 2nd power */
    r__15 = x[6] - x[3];
    /* Computing 2nd power */
    r__16 = x[5] - x[2];
    /* Computing 2nd power */
    r__17   = x[4] - x[1];
    h__[25] = -(*k * (sqrt(r__1 * r__1 + r__2 * r__2 + r__3 * r__3) - *l0)) / (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) * *l0) + r__7 * r__7 * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__11 * r__11 + r__12 * r__12 + r__13 * r__13) - *l0) / *l0 - r__14 * r__14 * *k / ((r__15 * r__15 + r__16 * r__16 + r__17 * r__17) * *l0);
    /* Computing 2nd power */
    r__1 = x[6] - x[3];
    /* Computing 2nd power */
    r__2 = x[5] - x[2];
    /* Computing 2nd power */
    r__3 = x[4] - x[1];
    d__1 = (T)(r__1 * r__1 + r__2 * r__2 + r__3 * r__3);
    /* Computing 2nd power */
    r__4 = x[6] - x[3];
    /* Computing 2nd power */
    r__5 = x[5] - x[2];
    /* Computing 2nd power */
    r__6 = x[4] - x[1];
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[31] = (x[4] - x[1]) * (x[5] - x[2]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) - *l0) / *l0 - (x[4] - x[1]) * (x[5] - x[2]) * *k / ((r__7 * r__7 + r__8 * r__8 + r__9 * r__9) * *l0);
    /* Computing 2nd power */
    r__1 = x[6] - x[3];
    /* Computing 2nd power */
    r__2 = x[5] - x[2];
    /* Computing 2nd power */
    r__3 = x[4] - x[1];
    d__1 = (T)(r__1 * r__1 + r__2 * r__2 + r__3 * r__3);
    /* Computing 2nd power */
    r__4 = x[6] - x[3];
    /* Computing 2nd power */
    r__5 = x[5] - x[2];
    /* Computing 2nd power */
    r__6 = x[4] - x[1];
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[37] = (x[4] - x[1]) * (x[6] - x[3]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) - *l0) / *l0 - (x[4] - x[1]) * (x[6] - x[3]) * *k / ((r__7 * r__7 + r__8 * r__8 + r__9 * r__9) * *l0);
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
    d__1 = (T)(r__4 * r__4 + r__5 * r__5 + r__6 * r__6);
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9   = x[4] - x[1];
    h__[8] = (x[4] - x[1]) * (x[5] - x[2]) * *k / ((r__1 * r__1 + r__2 * r__2 + r__3 * r__3) * *l0) - (x[4] - x[1]) * (x[5] - x[2]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__7 * r__7 + r__8 * r__8 + r__9 * r__9) - *l0) / *l0;
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
    /* Computing 2nd power */
    r__7 = x[5] - x[2];
    /* Computing 2nd power */
    r__8 = x[6] - x[3];
    /* Computing 2nd power */
    r__9 = x[5] - x[2];
    /* Computing 2nd power */
    r__10 = x[4] - x[1];
    d__1  = (T)(r__8 * r__8 + r__9 * r__9 + r__10 * r__10);
    /* Computing 2nd power */
    r__11 = x[6] - x[3];
    /* Computing 2nd power */
    r__12 = x[5] - x[2];
    /* Computing 2nd power */
    r__13 = x[4] - x[1];
    /* Computing 2nd power */
    r__14 = x[5] - x[2];
    /* Computing 2nd power */
    r__15 = x[6] - x[3];
    /* Computing 2nd power */
    r__16 = x[5] - x[2];
    /* Computing 2nd power */
    r__17   = x[4] - x[1];
    h__[14] = *k * (sqrt(r__1 * r__1 + r__2 * r__2 + r__3 * r__3) - *l0) / (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) * *l0) - r__7 * r__7 * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__11 * r__11 + r__12 * r__12 + r__13 * r__13) - *l0) / *l0 + r__14 * r__14 * *k / ((r__15 * r__15 + r__16 * r__16 + r__17 * r__17) * *l0);
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
    d__1 = (T)(r__4 * r__4 + r__5 * r__5 + r__6 * r__6);
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[20] = (x[5] - x[2]) * (x[6] - x[3]) * *k / ((r__1 * r__1 + r__2 * r__2 + r__3 * r__3) * *l0) - (x[5] - x[2]) * (x[6] - x[3]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__7 * r__7 + r__8 * r__8 + r__9 * r__9) - *l0) / *l0;
    /* Computing 2nd power */
    r__1 = x[6] - x[3];
    /* Computing 2nd power */
    r__2 = x[5] - x[2];
    /* Computing 2nd power */
    r__3 = x[4] - x[1];
    d__1 = (T)(r__1 * r__1 + r__2 * r__2 + r__3 * r__3);
    /* Computing 2nd power */
    r__4 = x[6] - x[3];
    /* Computing 2nd power */
    r__5 = x[5] - x[2];
    /* Computing 2nd power */
    r__6 = x[4] - x[1];
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[26] = (x[4] - x[1]) * (x[5] - x[2]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) - *l0) / *l0 - (x[4] - x[1]) * (x[5] - x[2]) * *k / ((r__7 * r__7 + r__8 * r__8 + r__9 * r__9) * *l0);
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
    /* Computing 2nd power */
    r__7 = x[5] - x[2];
    /* Computing 2nd power */
    r__8 = x[6] - x[3];
    /* Computing 2nd power */
    r__9 = x[5] - x[2];
    /* Computing 2nd power */
    r__10 = x[4] - x[1];
    d__1  = (T)(r__8 * r__8 + r__9 * r__9 + r__10 * r__10);
    /* Computing 2nd power */
    r__11 = x[6] - x[3];
    /* Computing 2nd power */
    r__12 = x[5] - x[2];
    /* Computing 2nd power */
    r__13 = x[4] - x[1];
    /* Computing 2nd power */
    r__14 = x[5] - x[2];
    /* Computing 2nd power */
    r__15 = x[6] - x[3];
    /* Computing 2nd power */
    r__16 = x[5] - x[2];
    /* Computing 2nd power */
    r__17   = x[4] - x[1];
    h__[32] = -(*k * (sqrt(r__1 * r__1 + r__2 * r__2 + r__3 * r__3) - *l0)) / (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) * *l0) + r__7 * r__7 * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__11 * r__11 + r__12 * r__12 + r__13 * r__13) - *l0) / *l0 - r__14 * r__14 * *k / ((r__15 * r__15 + r__16 * r__16 + r__17 * r__17) * *l0);
    /* Computing 2nd power */
    r__1 = x[6] - x[3];
    /* Computing 2nd power */
    r__2 = x[5] - x[2];
    /* Computing 2nd power */
    r__3 = x[4] - x[1];
    d__1 = (T)(r__1 * r__1 + r__2 * r__2 + r__3 * r__3);
    /* Computing 2nd power */
    r__4 = x[6] - x[3];
    /* Computing 2nd power */
    r__5 = x[5] - x[2];
    /* Computing 2nd power */
    r__6 = x[4] - x[1];
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[38] = (x[5] - x[2]) * (x[6] - x[3]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) - *l0) / *l0 - (x[5] - x[2]) * (x[6] - x[3]) * *k / ((r__7 * r__7 + r__8 * r__8 + r__9 * r__9) * *l0);
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
    d__1 = (T)(r__4 * r__4 + r__5 * r__5 + r__6 * r__6);
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9   = x[4] - x[1];
    h__[9] = (x[4] - x[1]) * (x[6] - x[3]) * *k / ((r__1 * r__1 + r__2 * r__2 + r__3 * r__3) * *l0) - (x[4] - x[1]) * (x[6] - x[3]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__7 * r__7 + r__8 * r__8 + r__9 * r__9) - *l0) / *l0;
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
    d__1 = (T)(r__4 * r__4 + r__5 * r__5 + r__6 * r__6);
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[15] = (x[5] - x[2]) * (x[6] - x[3]) * *k / ((r__1 * r__1 + r__2 * r__2 + r__3 * r__3) * *l0) - (x[5] - x[2]) * (x[6] - x[3]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__7 * r__7 + r__8 * r__8 + r__9 * r__9) - *l0) / *l0;
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
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[6] - x[3];
    /* Computing 2nd power */
    r__9 = x[5] - x[2];
    /* Computing 2nd power */
    r__10 = x[4] - x[1];
    d__1  = (T)(r__8 * r__8 + r__9 * r__9 + r__10 * r__10);
    /* Computing 2nd power */
    r__11 = x[6] - x[3];
    /* Computing 2nd power */
    r__12 = x[5] - x[2];
    /* Computing 2nd power */
    r__13 = x[4] - x[1];
    /* Computing 2nd power */
    r__14 = x[6] - x[3];
    /* Computing 2nd power */
    r__15 = x[6] - x[3];
    /* Computing 2nd power */
    r__16 = x[5] - x[2];
    /* Computing 2nd power */
    r__17   = x[4] - x[1];
    h__[21] = *k * (sqrt(r__1 * r__1 + r__2 * r__2 + r__3 * r__3) - *l0) / (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) * *l0) - r__7 * r__7 * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__11 * r__11 + r__12 * r__12 + r__13 * r__13) - *l0) / *l0 + r__14 * r__14 * *k / ((r__15 * r__15 + r__16 * r__16 + r__17 * r__17) * *l0);
    /* Computing 2nd power */
    r__1 = x[6] - x[3];
    /* Computing 2nd power */
    r__2 = x[5] - x[2];
    /* Computing 2nd power */
    r__3 = x[4] - x[1];
    d__1 = (T)(r__1 * r__1 + r__2 * r__2 + r__3 * r__3);
    /* Computing 2nd power */
    r__4 = x[6] - x[3];
    /* Computing 2nd power */
    r__5 = x[5] - x[2];
    /* Computing 2nd power */
    r__6 = x[4] - x[1];
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[27] = (x[4] - x[1]) * (x[6] - x[3]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) - *l0) / *l0 - (x[4] - x[1]) * (x[6] - x[3]) * *k / ((r__7 * r__7 + r__8 * r__8 + r__9 * r__9) * *l0);
    /* Computing 2nd power */
    r__1 = x[6] - x[3];
    /* Computing 2nd power */
    r__2 = x[5] - x[2];
    /* Computing 2nd power */
    r__3 = x[4] - x[1];
    d__1 = (T)(r__1 * r__1 + r__2 * r__2 + r__3 * r__3);
    /* Computing 2nd power */
    r__4 = x[6] - x[3];
    /* Computing 2nd power */
    r__5 = x[5] - x[2];
    /* Computing 2nd power */
    r__6 = x[4] - x[1];
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[33] = (x[5] - x[2]) * (x[6] - x[3]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) - *l0) / *l0 - (x[5] - x[2]) * (x[6] - x[3]) * *k / ((r__7 * r__7 + r__8 * r__8 + r__9 * r__9) * *l0);
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
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[6] - x[3];
    /* Computing 2nd power */
    r__9 = x[5] - x[2];
    /* Computing 2nd power */
    r__10 = x[4] - x[1];
    d__1  = (T)(r__8 * r__8 + r__9 * r__9 + r__10 * r__10);
    /* Computing 2nd power */
    r__11 = x[6] - x[3];
    /* Computing 2nd power */
    r__12 = x[5] - x[2];
    /* Computing 2nd power */
    r__13 = x[4] - x[1];
    /* Computing 2nd power */
    r__14 = x[6] - x[3];
    /* Computing 2nd power */
    r__15 = x[6] - x[3];
    /* Computing 2nd power */
    r__16 = x[5] - x[2];
    /* Computing 2nd power */
    r__17   = x[4] - x[1];
    h__[39] = -(*k * (sqrt(r__1 * r__1 + r__2 * r__2 + r__3 * r__3) - *l0)) / (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) * *l0) + r__7 * r__7 * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__11 * r__11 + r__12 * r__12 + r__13 * r__13) - *l0) / *l0 - r__14 * r__14 * *k / ((r__15 * r__15 + r__16 * r__16 + r__17 * r__17) * *l0);
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
    /* Computing 2nd power */
    r__7 = x[4] - x[1];
    /* Computing 2nd power */
    r__8 = x[6] - x[3];
    /* Computing 2nd power */
    r__9 = x[5] - x[2];
    /* Computing 2nd power */
    r__10 = x[4] - x[1];
    d__1  = (T)(r__8 * r__8 + r__9 * r__9 + r__10 * r__10);
    /* Computing 2nd power */
    r__11 = x[6] - x[3];
    /* Computing 2nd power */
    r__12 = x[5] - x[2];
    /* Computing 2nd power */
    r__13 = x[4] - x[1];
    /* Computing 2nd power */
    r__14 = x[4] - x[1];
    /* Computing 2nd power */
    r__15 = x[6] - x[3];
    /* Computing 2nd power */
    r__16 = x[5] - x[2];
    /* Computing 2nd power */
    r__17   = x[4] - x[1];
    h__[10] = -(*k * (sqrt(r__1 * r__1 + r__2 * r__2 + r__3 * r__3) - *l0)) / (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) * *l0) + r__7 * r__7 * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__11 * r__11 + r__12 * r__12 + r__13 * r__13) - *l0) / *l0 - r__14 * r__14 * *k / ((r__15 * r__15 + r__16 * r__16 + r__17 * r__17) * *l0);
    /* Computing 2nd power */
    r__1 = x[6] - x[3];
    /* Computing 2nd power */
    r__2 = x[5] - x[2];
    /* Computing 2nd power */
    r__3 = x[4] - x[1];
    d__1 = (T)(r__1 * r__1 + r__2 * r__2 + r__3 * r__3);
    /* Computing 2nd power */
    r__4 = x[6] - x[3];
    /* Computing 2nd power */
    r__5 = x[5] - x[2];
    /* Computing 2nd power */
    r__6 = x[4] - x[1];
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[16] = (x[4] - x[1]) * (x[5] - x[2]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) - *l0) / *l0 - (x[4] - x[1]) * (x[5] - x[2]) * *k / ((r__7 * r__7 + r__8 * r__8 + r__9 * r__9) * *l0);
    /* Computing 2nd power */
    r__1 = x[6] - x[3];
    /* Computing 2nd power */
    r__2 = x[5] - x[2];
    /* Computing 2nd power */
    r__3 = x[4] - x[1];
    d__1 = (T)(r__1 * r__1 + r__2 * r__2 + r__3 * r__3);
    /* Computing 2nd power */
    r__4 = x[6] - x[3];
    /* Computing 2nd power */
    r__5 = x[5] - x[2];
    /* Computing 2nd power */
    r__6 = x[4] - x[1];
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[22] = (x[4] - x[1]) * (x[6] - x[3]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) - *l0) / *l0 - (x[4] - x[1]) * (x[6] - x[3]) * *k / ((r__7 * r__7 + r__8 * r__8 + r__9 * r__9) * *l0);
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
    /* Computing 2nd power */
    r__7 = x[4] - x[1];
    /* Computing 2nd power */
    r__8 = x[6] - x[3];
    /* Computing 2nd power */
    r__9 = x[5] - x[2];
    /* Computing 2nd power */
    r__10 = x[4] - x[1];
    d__1  = (T)(r__8 * r__8 + r__9 * r__9 + r__10 * r__10);
    /* Computing 2nd power */
    r__11 = x[6] - x[3];
    /* Computing 2nd power */
    r__12 = x[5] - x[2];
    /* Computing 2nd power */
    r__13 = x[4] - x[1];
    /* Computing 2nd power */
    r__14 = x[4] - x[1];
    /* Computing 2nd power */
    r__15 = x[6] - x[3];
    /* Computing 2nd power */
    r__16 = x[5] - x[2];
    /* Computing 2nd power */
    r__17   = x[4] - x[1];
    h__[28] = *k * (sqrt(r__1 * r__1 + r__2 * r__2 + r__3 * r__3) - *l0) / (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) * *l0) - r__7 * r__7 * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__11 * r__11 + r__12 * r__12 + r__13 * r__13) - *l0) / *l0 + r__14 * r__14 * *k / ((r__15 * r__15 + r__16 * r__16 + r__17 * r__17) * *l0);
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
    d__1 = (T)(r__4 * r__4 + r__5 * r__5 + r__6 * r__6);
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[34] = (x[4] - x[1]) * (x[5] - x[2]) * *k / ((r__1 * r__1 + r__2 * r__2 + r__3 * r__3) * *l0) - (x[4] - x[1]) * (x[5] - x[2]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__7 * r__7 + r__8 * r__8 + r__9 * r__9) - *l0) / *l0;
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
    d__1 = (T)(r__4 * r__4 + r__5 * r__5 + r__6 * r__6);
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[40] = (x[4] - x[1]) * (x[6] - x[3]) * *k / ((r__1 * r__1 + r__2 * r__2 + r__3 * r__3) * *l0) - (x[4] - x[1]) * (x[6] - x[3]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__7 * r__7 + r__8 * r__8 + r__9 * r__9) - *l0) / *l0;
    /* Computing 2nd power */
    r__1 = x[6] - x[3];
    /* Computing 2nd power */
    r__2 = x[5] - x[2];
    /* Computing 2nd power */
    r__3 = x[4] - x[1];
    d__1 = (T)(r__1 * r__1 + r__2 * r__2 + r__3 * r__3);
    /* Computing 2nd power */
    r__4 = x[6] - x[3];
    /* Computing 2nd power */
    r__5 = x[5] - x[2];
    /* Computing 2nd power */
    r__6 = x[4] - x[1];
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[11] = (x[4] - x[1]) * (x[5] - x[2]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) - *l0) / *l0 - (x[4] - x[1]) * (x[5] - x[2]) * *k / ((r__7 * r__7 + r__8 * r__8 + r__9 * r__9) * *l0);
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
    /* Computing 2nd power */
    r__7 = x[5] - x[2];
    /* Computing 2nd power */
    r__8 = x[6] - x[3];
    /* Computing 2nd power */
    r__9 = x[5] - x[2];
    /* Computing 2nd power */
    r__10 = x[4] - x[1];
    d__1  = (T)(r__8 * r__8 + r__9 * r__9 + r__10 * r__10);
    /* Computing 2nd power */
    r__11 = x[6] - x[3];
    /* Computing 2nd power */
    r__12 = x[5] - x[2];
    /* Computing 2nd power */
    r__13 = x[4] - x[1];
    /* Computing 2nd power */
    r__14 = x[5] - x[2];
    /* Computing 2nd power */
    r__15 = x[6] - x[3];
    /* Computing 2nd power */
    r__16 = x[5] - x[2];
    /* Computing 2nd power */
    r__17   = x[4] - x[1];
    h__[17] = -(*k * (sqrt(r__1 * r__1 + r__2 * r__2 + r__3 * r__3) - *l0)) / (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) * *l0) + r__7 * r__7 * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__11 * r__11 + r__12 * r__12 + r__13 * r__13) - *l0) / *l0 - r__14 * r__14 * *k / ((r__15 * r__15 + r__16 * r__16 + r__17 * r__17) * *l0);
    /* Computing 2nd power */
    r__1 = x[6] - x[3];
    /* Computing 2nd power */
    r__2 = x[5] - x[2];
    /* Computing 2nd power */
    r__3 = x[4] - x[1];
    d__1 = (T)(r__1 * r__1 + r__2 * r__2 + r__3 * r__3);
    /* Computing 2nd power */
    r__4 = x[6] - x[3];
    /* Computing 2nd power */
    r__5 = x[5] - x[2];
    /* Computing 2nd power */
    r__6 = x[4] - x[1];
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[23] = (x[5] - x[2]) * (x[6] - x[3]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) - *l0) / *l0 - (x[5] - x[2]) * (x[6] - x[3]) * *k / ((r__7 * r__7 + r__8 * r__8 + r__9 * r__9) * *l0);
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
    d__1 = (T)(r__4 * r__4 + r__5 * r__5 + r__6 * r__6);
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[29] = (x[4] - x[1]) * (x[5] - x[2]) * *k / ((r__1 * r__1 + r__2 * r__2 + r__3 * r__3) * *l0) - (x[4] - x[1]) * (x[5] - x[2]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__7 * r__7 + r__8 * r__8 + r__9 * r__9) - *l0) / *l0;
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
    /* Computing 2nd power */
    r__7 = x[5] - x[2];
    /* Computing 2nd power */
    r__8 = x[6] - x[3];
    /* Computing 2nd power */
    r__9 = x[5] - x[2];
    /* Computing 2nd power */
    r__10 = x[4] - x[1];
    d__1  = (T)(r__8 * r__8 + r__9 * r__9 + r__10 * r__10);
    /* Computing 2nd power */
    r__11 = x[6] - x[3];
    /* Computing 2nd power */
    r__12 = x[5] - x[2];
    /* Computing 2nd power */
    r__13 = x[4] - x[1];
    /* Computing 2nd power */
    r__14 = x[5] - x[2];
    /* Computing 2nd power */
    r__15 = x[6] - x[3];
    /* Computing 2nd power */
    r__16 = x[5] - x[2];
    /* Computing 2nd power */
    r__17   = x[4] - x[1];
    h__[35] = *k * (sqrt(r__1 * r__1 + r__2 * r__2 + r__3 * r__3) - *l0) / (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) * *l0) - r__7 * r__7 * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__11 * r__11 + r__12 * r__12 + r__13 * r__13) - *l0) / *l0 + r__14 * r__14 * *k / ((r__15 * r__15 + r__16 * r__16 + r__17 * r__17) * *l0);
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
    d__1 = (T)(r__4 * r__4 + r__5 * r__5 + r__6 * r__6);
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[41] = (x[5] - x[2]) * (x[6] - x[3]) * *k / ((r__1 * r__1 + r__2 * r__2 + r__3 * r__3) * *l0) - (x[5] - x[2]) * (x[6] - x[3]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__7 * r__7 + r__8 * r__8 + r__9 * r__9) - *l0) / *l0;
    /* Computing 2nd power */
    r__1 = x[6] - x[3];
    /* Computing 2nd power */
    r__2 = x[5] - x[2];
    /* Computing 2nd power */
    r__3 = x[4] - x[1];
    d__1 = (T)(r__1 * r__1 + r__2 * r__2 + r__3 * r__3);
    /* Computing 2nd power */
    r__4 = x[6] - x[3];
    /* Computing 2nd power */
    r__5 = x[5] - x[2];
    /* Computing 2nd power */
    r__6 = x[4] - x[1];
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[12] = (x[4] - x[1]) * (x[6] - x[3]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) - *l0) / *l0 - (x[4] - x[1]) * (x[6] - x[3]) * *k / ((r__7 * r__7 + r__8 * r__8 + r__9 * r__9) * *l0);
    /* Computing 2nd power */
    r__1 = x[6] - x[3];
    /* Computing 2nd power */
    r__2 = x[5] - x[2];
    /* Computing 2nd power */
    r__3 = x[4] - x[1];
    d__1 = (T)(r__1 * r__1 + r__2 * r__2 + r__3 * r__3);
    /* Computing 2nd power */
    r__4 = x[6] - x[3];
    /* Computing 2nd power */
    r__5 = x[5] - x[2];
    /* Computing 2nd power */
    r__6 = x[4] - x[1];
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[18] = (x[5] - x[2]) * (x[6] - x[3]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) - *l0) / *l0 - (x[5] - x[2]) * (x[6] - x[3]) * *k / ((r__7 * r__7 + r__8 * r__8 + r__9 * r__9) * *l0);
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
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[6] - x[3];
    /* Computing 2nd power */
    r__9 = x[5] - x[2];
    /* Computing 2nd power */
    r__10 = x[4] - x[1];
    d__1  = (T)(r__8 * r__8 + r__9 * r__9 + r__10 * r__10);
    /* Computing 2nd power */
    r__11 = x[6] - x[3];
    /* Computing 2nd power */
    r__12 = x[5] - x[2];
    /* Computing 2nd power */
    r__13 = x[4] - x[1];
    /* Computing 2nd power */
    r__14 = x[6] - x[3];
    /* Computing 2nd power */
    r__15 = x[6] - x[3];
    /* Computing 2nd power */
    r__16 = x[5] - x[2];
    /* Computing 2nd power */
    r__17   = x[4] - x[1];
    h__[24] = -(*k * (sqrt(r__1 * r__1 + r__2 * r__2 + r__3 * r__3) - *l0)) / (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) * *l0) + r__7 * r__7 * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__11 * r__11 + r__12 * r__12 + r__13 * r__13) - *l0) / *l0 - r__14 * r__14 * *k / ((r__15 * r__15 + r__16 * r__16 + r__17 * r__17) * *l0);
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
    d__1 = (T)(r__4 * r__4 + r__5 * r__5 + r__6 * r__6);
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[30] = (x[4] - x[1]) * (x[6] - x[3]) * *k / ((r__1 * r__1 + r__2 * r__2 + r__3 * r__3) * *l0) - (x[4] - x[1]) * (x[6] - x[3]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__7 * r__7 + r__8 * r__8 + r__9 * r__9) - *l0) / *l0;
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
    d__1 = (T)(r__4 * r__4 + r__5 * r__5 + r__6 * r__6);
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[5] - x[2];
    /* Computing 2nd power */
    r__9    = x[4] - x[1];
    h__[36] = (x[5] - x[2]) * (x[6] - x[3]) * *k / ((r__1 * r__1 + r__2 * r__2 + r__3 * r__3) * *l0) - (x[5] - x[2]) * (x[6] - x[3]) * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__7 * r__7 + r__8 * r__8 + r__9 * r__9) - *l0) / *l0;
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
    /* Computing 2nd power */
    r__7 = x[6] - x[3];
    /* Computing 2nd power */
    r__8 = x[6] - x[3];
    /* Computing 2nd power */
    r__9 = x[5] - x[2];
    /* Computing 2nd power */
    r__10 = x[4] - x[1];
    d__1  = (T)(r__8 * r__8 + r__9 * r__9 + r__10 * r__10);
    /* Computing 2nd power */
    r__11 = x[6] - x[3];
    /* Computing 2nd power */
    r__12 = x[5] - x[2];
    /* Computing 2nd power */
    r__13 = x[4] - x[1];
    /* Computing 2nd power */
    r__14 = x[6] - x[3];
    /* Computing 2nd power */
    r__15 = x[6] - x[3];
    /* Computing 2nd power */
    r__16 = x[5] - x[2];
    /* Computing 2nd power */
    r__17   = x[4] - x[1];
    h__[42] = *k * (sqrt(r__1 * r__1 + r__2 * r__2 + r__3 * r__3) - *l0) / (sqrt(r__4 * r__4 + r__5 * r__5 + r__6 * r__6) * *l0) - r__7 * r__7 * pow_dd(&d__1, &c_b2) * *k * (sqrt(r__11 * r__11 + r__12 * r__12 + r__13 * r__13) - *l0) / *l0 + r__14 * r__14 * *k / ((r__15 * r__15 + r__16 * r__16 + r__17 * r__17) * *l0);
    return 0;
} /* edge_hessian__ */

template int EdgeHessian<double>(double* __restrict x, double* __restrict l0, double* __restrict k, double* __restrict h__);
template int EdgeHessian<float>(float* __restrict x, float* __restrict l0, float* __restrict k, float* __restrict h__);

template double pow_dd<double>(double* __restrict b, double* __restrict a);
template float  pow_dd<float>(float* __restrict b, float* __restrict a);

#endif  // EDGE_HESSIAN_JJ_H
