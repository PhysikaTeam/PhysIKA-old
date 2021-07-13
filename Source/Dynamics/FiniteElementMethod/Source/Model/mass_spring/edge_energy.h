/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: edge energy for mass spring method.
 * @version    : 1.0
 */
#ifndef EDGE_ENERGY_JJ_H
#define EDGE_ENERGY_JJ_H

/* edge_energy.f -- translated by f2c (version 20160102).
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
/* Subroutine */ int EdgeEnergy(T* __restrict x, T* __restrict k, T* __restrict l0, T* __restrict v)
{
    /* System generated locals */
    T r__1, r__2, r__3, r__4;

    /* Builtin functions */
    /* Parameter adjustments */
    --x;

    /* Function Body */
    /* Computing 2nd power */
    r__2 = x[6] - x[3];
    /* Computing 2nd power */
    r__3 = x[5] - x[2];
    /* Computing 2nd power */
    r__4 = x[4] - x[1];
    /* Computing 2nd power */
    r__1 = sqrt(r__2 * r__2 + r__3 * r__3 + r__4 * r__4) - *l0;
    *v   = *k * (r__1 * r__1) / *l0 / 2.f;
    return 0;
} /* edge_energy__ */

template int EdgeEnergy<double>(double* __restrict x, double* __restrict l0, double* __restrict k, double* __restrict v);
template int EdgeEnergy<float>(float* __restrict x, float* __restrict l0, float* __restrict k, float* __restrict v);

#endif  // EDGE_ENERGY_JJ_H
