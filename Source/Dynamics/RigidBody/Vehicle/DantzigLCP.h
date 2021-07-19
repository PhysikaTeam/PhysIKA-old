/*************************************************************************
 *                                                                       *
 * Open Dynamics Engine, Copyright (C) 2001,2002 Russell L. Smith.       *
 * All rights reserved.  Email: russ@q12.org   Web: www.q12.org          *
 *                                                                       *
 * This library is free software; you can redistribute it and/or         *
 * modify it under the terms of                                          * 
 *   The BSD-style license that is included with this library in         *
 *   the file LICENSE-BSD.TXT.                                           *
 *                                                                       *
 * This library is distributed in the hope that it will be useful,       *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the files    *
 * LICENSE.TXT and LICENSE-BSD.TXT for more details.                     *
 *                                                                       *
 *************************************************************************/

/*

given (A,b,lo,hi), solve the LCP problem: A*x = b+w, where each x(i),w(i)
satisfies one of
    (1) x = lo, w >= 0
    (2) x = hi, w <= 0
    (3) lo < x < hi, w = 0
A is a matrix of dimension n*n, everything else is a vector of size n*1.
lo and hi can be +/- dInfinity as needed. the first `nub' variables are
unbounded, i.e. hi and lo are assumed to be +/- dInfinity.

we restrict lo(i) <= 0 and hi(i) >= 0.

the original data (A,b) may be modified by this function.

if the `findex' (friction index) parameter is nonzero, it points to an array
of index values. in this case constraints that have findex[i] >= 0 are
special. all non-special constraints are solved for, then the lo and hi values
for the special constraints are set:
  hi[i] = abs( hi[i] * x[findex[i]] )
  lo[i] = -hi[i]
and the solution continues. this mechanism allows a friction approximation
to be implemented. the first `nub' variables are assumed to have findex < 0.

*/

#ifndef _DANTZIGLCP_H_
#define _DANTZIGLCP_H_

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "Core/Array/Array.h"
#include <iostream>

#ifndef BT_NAN
static int btNanMask = 0x7F800001;
#define BT_NAN (*( float* )&btNanMask)
#endif

#ifndef BT_INFINITY
static int btInfinityMask = 0x7F800000;
#define BT_INFINITY (*( float* )&btInfinityMask)
inline int btGetInfinityMask()  //suppress stupid compiler warning
{
    return btInfinityMask;
}
#endif

namespace PhysIKA {

struct DantzigScratchMemory
{
    HostArray<float>  m_scratch;
    HostArray<float>  L;
    HostArray<float>  d;
    HostArray<float>  delta_w;
    HostArray<float>  delta_x;
    HostArray<float>  Dell;
    HostArray<float>  ell;
    HostArray<float*> Arows;
    HostArray<int>    p;
    HostArray<int>    C;
    HostArray<bool>   state;
};

struct DantzigInputMemory
{
    HostArray<float> A;
    HostArray<float> x;
    HostArray<float> b;
    HostArray<float> w;
    HostArray<float> lo;
    HostArray<float> hi;
    HostArray<int>   findex;

    void resize(int n)
    {

        std::cout << " In DantzigInputMemory:  OK" << std::endl;
        A.resize(n * n);
        x.resize(n);
        b.resize(n);
        w.resize(n);
        lo.resize(n);
        hi.resize(n);
        findex.resize(n);
    }
};

//return false if solving failed
bool btSolveDantzigLCP(int n, float* A, float* x, float* b, float* w, int nub, float* lo, float* hi, int* findex, DantzigScratchMemory& scratch);
}  // namespace PhysIKA

#endif  //_DANTZIGLCP_H_
