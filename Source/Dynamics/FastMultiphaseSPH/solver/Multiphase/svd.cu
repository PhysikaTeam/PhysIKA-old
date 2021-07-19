/**
Copyright (c) 2016 Theodore Gast, Chuyuan Fu, Chenfanfu Jiang, Joseph Teran

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

If the code is used in an article, the following paper shall be cited:
@techreport{qrsvd:2016,
title={Implicit-shifted Symmetric QR Singular Value Decomposition of 3x3 Matrices},
author={Gast, Theodore and Fu, Chuyuan and Jiang, Chenfanfu and Teran, Joseph},
year={2016},
institution={University of California Los Angeles}
}

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

################################################################################
This file implements 2D and 3D polar decompositions and SVDs.

float may be float or double.

2D Polar:
Eigen::Matrix<float, 2, 2> A,R,S;
A<<1,2,3,4;
JIXIE::polarDecomposition(A, R, S);
// R will be the closest rotation to A
// S will be symmetric

2D SVD:
Eigen::Matrix<float, 2, 2> A;
A<<1,2,3,4;
Eigen::Matrix<float, 2, 1> S;
Eigen::Matrix<float, 2, 2> U;
Eigen::Matrix<float, 2, 2> V;
JIXIE::singularValueDecomposition(A,U,S,V);
// A = U S V'
// U and V will be rotations
// S will be singular values sorted by decreasing magnitude. Only the last one may be negative.

3D Polar:
Eigen::Matrix<float, 3, 3> A,R,S;
A<<1,2,3,4,5,6;
JIXIE::polarDecomposition(A, R, S);
// R will be the closest rotation to A
// S will be symmetric

3D SVD:
Eigen::Matrix<float, 3, 3> A;
A<<1,2,3,4,5,6;
Eigen::Matrix<float, 3, 1> S;
Eigen::Matrix<float, 3, 3> U;
Eigen::Matrix<float, 3, 3> V;
JIXIE::singularValueDecomposition(A,U,S,V);
// A = U S V'
// U and V will be rotations
// S will be singular values sorted by decreasing magnitude. Only the last one may be negative.

################################################################################
*/

/**
SVD based on implicit QR with Wilkinson Shift
*/

/**
Class for givens rotation.
Row rotation G*A corresponds to something like
c -s  0
( s  c  0 ) A
0  0  1
Column rotation A G' corresponds to something like
c -s  0
A ( s  c  0 )
0  0  1

c and s are always computed so that
( c -s ) ( a )  =  ( * )
s  c     b       ( 0 )

Assume rowi<rowk.
*/

#include "svd.h"

HDFUNC inline void cswap(float& x, float& y)
{
    float temp = x;
    x          = y;
    y          = temp;
}

HDFUNC inline void zeroChase(cmat3& H, cmat3& U, cmat3& V)
{
    GivensRotation r1(H[0][0], H[1][0], 0, 1);
    GivensRotation r2(1, 2);
    if (!ZERO(H[1][0]))
        r2.compute(H[0][0] * H[0][1] + H[1][0] * H[1][1], H[0][0] * H[0][2] + H[1][0] * H[1][2]);
    else
        r2.compute(H[0][1], H[0][2]);

    r1.rowRotation3(H);

    /* GivensRotation<float> r2(H(0, 1), H(0, 2), 1, 2); */
    r2.columnRotation3(H);
    r2.columnRotation3(V);

    /**
	Reduce H to of form
	x x 0
	0 x x
	0 0 x
	*/
    GivensRotation r3(H[1][1], H[2][1], 1, 2);
    r3.rowRotation3(H);

    // Save this till end for better cache coherency
    // r1.rowRotation(u_transpose);
    // r3.rowRotation(u_transpose);
    r1.columnRotation3(U);
    r3.columnRotation3(U);
}

/**
\brief make a 3X3 matrix to upper bidiagonal form
original form of H:   x x x
x x x
x x x
after zero chase:
x x 0
0 x x
0 0 x
*/
HDFUNC inline void makeUpperBidiag(cmat3& H, cmat3& U, cmat3& V)
{
    make_identity(U);
    make_identity(V);

    /**
	Reduce H to of form
	x x x
	x x x
	0 x x
	*/

    GivensRotation r(H[1][0], H[2][0], 1, 2);
    r.rowRotation3(H);
    // r.rowRotation(u_transpose);
    r.columnRotation3(U);
    // zeroChase(H, u_transpose, V);
    zeroChase(H, U, V);
}

/**
\brief make a 3X3 matrix to lambda shape
original form of H:   x x x
*                     x x x
*                     x x x
after :
*                     x 0 0
*                     x x 0
*                     x 0 x
*/
HDFUNC inline void makeLambdaShape(cmat3& H, cmat3& U, cmat3& V)
{
    make_identity(U);
    make_identity(V);

    /**
	Reduce H to of form
	*                    x x 0
	*                    x x x
	*                    x x x
	*/

    GivensRotation r1(H[0][1], H[0][2], 1, 2);
    r1.columnRotation3(H);
    r1.columnRotation3(V);

    /**
	Reduce H to of form
	*                    x x 0
	*                    x x 0
	*                    x x x
	*/

    r1.computeUnconventional(H[1][2], H[2][2]);
    r1.rowRotation3(H);
    r1.columnRotation3(U);

    /**
	Reduce H to of form
	*                    x x 0
	*                    x x 0
	*                    x 0 x
	*/

    GivensRotation r2(H[2][0], H[2][1], 0, 1);
    r2.columnRotation3(H);
    r2.columnRotation3(V);

    /**
	Reduce H to of form
	*                    x 0 0
	*                    x x 0
	*                    x 0 x
	*/
    r2.computeUnconventional(H[0][1], H[1][1]);
    r2.rowRotation3(H);
    r2.columnRotation3(U);
}

/**
\brief 2x2 polar decomposition.
\param[in] A matrix.
\param[out] R Robustly a rotation matrix in givens form
\param[out] S_Sym Symmetric. Whole matrix is stored

Whole matrix S is stored since its faster to calculate due to simd vectorization
Polar guarantees negative sign is on the small magnitude singular value.
S is guaranteed to be the closest one to identity.
R is guaranteed to be the closest rotation to A.
*/
HDFUNC inline void polarDecomposition2(cmat2& A, GivensRotation& R, cmat2& S)
{
    float x0          = A[0][0] + A[1][1];
    float x1          = A[1][0] - A[0][1];
    float denominator = sqrt(x0 * x0 + x1 * x1);
    R.c               = ( float )1;
    R.s               = ( float )0;
    if (denominator != 0)
    {
        R.c = x0 / denominator;
        R.s = -x1 / denominator;
    }
    S = A;
    R.rowRotation2(S);
}

/**
\brief 2x2 polar decomposition.
\param[in] A matrix.
\param[out] R Robustly a rotation matrix.
\param[out] S_Sym Symmetric. Whole matrix is stored

Whole matrix S is stored since its faster to calculate due to simd vectorization
Polar guarantees negative sign is on the small magnitude singular value.
S is guaranteed to be the closest one to identity.
R is guaranteed to be the closest rotation to A.
*/
HDFUNC inline void polarDecomposition2(cmat2& A, cmat2& R, cmat2& S)
{
    GivensRotation r(0, 1);
    polarDecomposition2(A, r, S);
    r.fill2(R);
}

/**
\brief 2x2 SVD (singular value decomposition) A=USV'
\param[in] A Input matrix.
\param[out] U Robustly a rotation matrix in Givens form
\param[out] Sigma Vector of singular values sorted with decreasing magnitude. The second one can be negative.
\param[out] V Robustly a rotation matrix in Givens form
*/
HDFUNC inline void singularValueDecomposition2(cmat2& A, GivensRotation& U, cfloat2& Sigma, GivensRotation& V, const float tol = 64 * 1e-10)
{
    cmat2 S;
    polarDecomposition2(A, U, S);
    float cosine, sine;
    float x = S[0][0];
    float y = S[0][1];
    float z = S[1][1];
    if (ZERO(y))
    {
        cosine  = 1;
        sine    = 0;
        Sigma.x = x;
        Sigma.y = z;
    }
    else
    {
        float tau = 0.5 * (x - z);
        float w   = sqrt(tau * tau + y * y);
        // w > y > 0
        float t;
        if (tau > 0)
        {
            // tau + w > w > y > 0 ==> division is safe
            t = y / (tau + w);
        }
        else
        {
            // tau - w < -w < -y < 0 ==> division is safe
            t = y / (tau - w);
        }
        cosine = float(1) / sqrt(t * t + float(1));
        sine   = -t * cosine;
        /*
		V = [cosine -sine; sine cosine]
		Sigma = V'SV. Only compute the diagonals for efficiency.
		Also utilize symmetry of S and don't form V yet.
		*/
        float c2  = cosine * cosine;
        float csy = 2 * cosine * sine * y;
        float s2  = sine * sine;
        Sigma.x   = c2 * x - csy + s2 * z;
        Sigma.y   = s2 * x + csy + c2 * z;
    }

    // Sorting
    // Polar already guarantees negative sign is on the small magnitude singular value.
    if (Sigma.x < Sigma.y)
    {
        cswap(Sigma.x, Sigma.y);
        V.c = -sine;
        V.s = cosine;
    }
    else
    {
        V.c = cosine;
        V.s = sine;
    }
    U *= V;
}
/**
\brief 2x2 SVD (singular value decomposition) A=USV'
\param[in] A Input matrix.
\param[out] U Robustly a rotation matrix.
\param[out] Sigma Vector of singular values sorted with decreasing magnitude. The second one can be negative.
\param[out] V Robustly a rotation matrix.
*/
HDFUNC inline void singularValueDecomposition2(cmat2& A, cmat2& U, cfloat2& Sigma, cmat2& V, const float tol = 64 * 1e-10)
{
    GivensRotation gv(0, 1);
    GivensRotation gu(0, 1);
    singularValueDecomposition2(A, gu, Sigma, gv, tol);

    gu.fill2(U);
    gv.fill2(V);
}

/**
\brief compute wilkinsonShift of the block
a1     b1
b1     a2
based on the wilkinsonShift formula
mu = c + d - sign (d) \ sqrt (d*d + b*b), where d = (a-c)/2

*/
HDFUNC float wilkinsonShift(const float a1, const float b1, const float a2)
{
    float d  = ( float )0.5 * (a1 - a2);
    float bs = b1 * b1;
    float mu = a2 - copysign(bs / (fabs(d) + sqrt(d * d + bs)), d);
    // float mu = a2 - bs / ( d + sign_d*sqrt (d*d + bs));
    return mu;
}

/**
\brief Helper function of 3X3 SVD for processing 2X2 SVD
*/
HDFUNC inline void process(int t, cmat3& B, cmat3& U, cfloat3& sigma, cmat3& V)
{

    int            other = (t == 1) ? 0 : 2;
    GivensRotation u(0, 1);
    GivensRotation v(0, 1);

    sigma[other]   = B[other][other];
    cfloat2 sigma2 = other == 0 ? cfloat2(sigma[1], sigma[2]) : cfloat2(sigma[0], sigma[1]);

    cmat2 b2 = other == 0 ? cmat2(B[1][1], B[1][2], B[2][1], B[2][2]) : cmat2(B[0][0], B[0][1], B[1][0], B[1][1]);

    singularValueDecomposition2(b2, u, sigma2, v);

    if (other == 0)
    {
        B[1][1]  = b2[0][0];
        B[1][2]  = b2[0][1];
        B[2][1]  = b2[1][0];
        B[2][2]  = b2[1][1];
        sigma[1] = sigma2.x;
        sigma[2] = sigma2.y;
    }
    else
    {
        B[0][0]  = b2[0][0];
        B[0][1]  = b2[0][1];
        B[1][0]  = b2[1][0];
        B[1][1]  = b2[1][1];
        sigma[0] = sigma2.x;
        sigma[1] = sigma2.y;
    }

    u.rowi += t;
    u.rowk += t;
    v.rowi += t;
    v.rowk += t;
    u.columnRotation3(U);
    v.columnRotation3(V);
}

/**
\brief Helper function of 3X3 SVD for flipping signs due to flipping signs of sigma
*/
HDFUNC inline void flipSign(int i, cmat3& U, cfloat3& sigma)
{
    sigma[i] = -sigma[i];
    U[0][i]  = -U[0][i];
    U[1][i]  = -U[1][i];
    U[2][i]  = -U[2][i];
}

HDFUNC inline void colswap(cmat3& A, int c1, int c2)
{
    cswap(A[0][c1], A[0][c2]);
    cswap(A[1][c1], A[1][c2]);
    cswap(A[2][c1], A[2][c2]);
}

/**
\brief Helper function of 3X3 SVD for sorting singular values
*/
HDFUNC inline void sort0(cmat3& U, cfloat3& sigma, cmat3& V)
{
    // Case: sigma(0) > |sigma(1)| >= |sigma(2)|
    if (fabs(sigma[1]) >= fabs(sigma[2]))
    {
        if (sigma[1] < 0)
        {
            flipSign(1, U, sigma);
            flipSign(2, U, sigma);
        }
        return;
    }

    //fix sign of sigma for both cases
    if (sigma[2] < 0)
    {
        flipSign(1, U, sigma);
        flipSign(2, U, sigma);
    }

    //swap sigma(1) and sigma(2) for both cases
    cswap(sigma[1], sigma[2]);
    colswap(U, 1, 2);
    colswap(V, 1, 2);

    // Case: |sigma(2)| >= sigma(0) > |simga(1)|
    if (sigma[1] > sigma[0])
    {
        cswap(sigma[0], sigma[1]);
        colswap(U, 0, 1);
        colswap(V, 0, 1);
    }

    // Case: sigma(0) >= |sigma(2)| > |simga(1)|
    else
    {
        U[0][2] = -U[0][2];
        U[1][2] = -U[1][2];
        U[2][2] = -U[2][2];
        V[0][2] = -V[0][2];
        V[1][2] = -V[1][2];
        V[2][2] = -V[2][2];
    }
}

/**
\brief Helper function of 3X3 SVD for sorting singular values
*/
HDFUNC inline void sort1(cmat3& U, cfloat3& sigma, cmat3& V)
{
    // Case: |sigma(0)| >= sigma(1) > |sigma(2)|
    if (fabs(sigma[0]) >= sigma[1])
    {
        if (sigma[0] < 0)
        {
            flipSign(0, U, sigma);
            flipSign(2, U, sigma);
        }
        return;
    }

    //swap sigma(0) and sigma(1) for both cases
    cswap(sigma[0], sigma[1]);
    colswap(U, 0, 1);
    colswap(V, 0, 1);

    // Case: sigma(1) > |sigma(2)| >= |sigma(0)|
    if (fabs(sigma[1]) < fabs(sigma[2]))
    {
        cswap(sigma[1], sigma[2]);
        colswap(U, 1, 2);
        colswap(V, 1, 2);
    }

    // Case: sigma(1) >= |sigma(0)| > |sigma(2)|
    else
    {
        U[0][1] = -U[0][1];
        U[1][1] = -U[1][1];
        U[2][1] = -U[2][1];
        V[0][1] = -V[0][1];
        V[1][1] = -V[1][1];
        V[2][1] = -V[2][1];
    }

    // fix sign for both cases
    if (sigma[1] < 0)
    {
        flipSign(1, U, sigma);
        flipSign(2, U, sigma);
    }
}

HDFUNC float mycmax(float a, float b)
{
    return a > b ? a : b;
}

/**
\brief 3X3 SVD (singular value decomposition) A=USV'
\param[in] A Input matrix.
\param[out] U is a rotation matrix.
\param[out] sigma Diagonal matrix, sorted with decreasing magnitude. The third one can be negative.
\param[out] V is a rotation matrix.
*/
HDFUNC int singularValueDecomposition(cmat3& A, cmat3& U, cfloat3& sigma, cmat3& V, float tol)
{
    cmat3 B = A;
    make_identity(U);
    make_identity(V);

    makeUpperBidiag(B, U, V);

    int            count = 0;
    float          mu    = 0.0f;
    GivensRotation r(0, 1);

    float alpha_1 = B[0][0];
    float beta_1  = B[0][1];
    float alpha_2 = B[1][1];
    float alpha_3 = B[2][2];
    float beta_2  = B[1][2];
    float gamma_1 = alpha_1 * beta_1;
    float gamma_2 = alpha_2 * beta_2;
    tol *= mycmax(0.5 * sqrt(alpha_1 * alpha_1 + alpha_2 * alpha_2 + alpha_3 * alpha_3 + beta_1 * beta_1 + beta_2 * beta_2), 1);

    /**
	Do implicit shift QR until A^float A is block diagonal
	*/

    while (fabsf(beta_2) > tol && fabsf(beta_1) > tol && fabsf(alpha_1) > tol && fabsf(alpha_2) > tol && fabsf(alpha_3) > tol)
    {
        mu = wilkinsonShift(alpha_2 * alpha_2 + beta_1 * beta_1, gamma_2, alpha_3 * alpha_3 + beta_2 * beta_2);

        r.compute(alpha_1 * alpha_1 - mu, gamma_1);
        r.columnRotation3(B);

        r.columnRotation3(V);
        zeroChase(B, U, V);

        alpha_1 = B[0][0];
        beta_1  = B[0][1];
        alpha_2 = B[1][1];
        alpha_3 = B[2][2];
        beta_2  = B[1][2];
        gamma_1 = alpha_1 * beta_1;
        gamma_2 = alpha_2 * beta_2;
        count++;
    }

    /**
	Handle the cases of one of the alphas and betas being 0
	Sorted by ease of handling and then frequency
	of occurrence

	If B is of form
	x x 0
	0 x 0
	0 0 x
	*/
    if (fabs(beta_2) <= tol)
    {
        process(0, B, U, sigma, V);
        sort0(U, sigma, V);
    }
    /**
	If B is of form
	x 0 0
	0 x x
	0 0 x
	*/
    else if (fabs(beta_1) <= tol)
    {
        process(1, B, U, sigma, V);
        sort1(U, sigma, V);
    }
    /**
	If B is of form
	x x 0
	0 0 x
	0 0 x
	*/
    else if (fabs(alpha_2) <= tol)
    {
        /**
		Reduce B to
		x x 0
		0 0 0
		0 0 x
		*/
        GivensRotation r1(1, 2);
        r1.computeUnconventional(B[1][2], B[2][2]);
        r1.rowRotation3(B);
        r1.columnRotation3(U);

        process(0, B, U, sigma, V);
        sort0(U, sigma, V);
    }
    /**
	If B is of form
	x x 0
	0 x x
	0 0 0
	*/
    else if (fabs(alpha_3) <= tol)
    {
        /**
		Reduce B to
		x x +
		0 x 0
		0 0 0
		*/
        GivensRotation r1(1, 2);
        r1.compute(B[1][1], B[1][2]);
        r1.columnRotation3(B);
        r1.columnRotation3(V);
        /**
		Reduce B to
		x x 0
		+ x 0
		0 0 0
		*/
        GivensRotation r2(0, 2);
        r2.compute(B[0][0], B[0][2]);
        r2.columnRotation3(B);
        r2.columnRotation3(V);

        process(0, B, U, sigma, V);
        sort0(U, sigma, V);
    }
    /**
	If B is of form
	0 x 0
	0 x x
	0 0 x
	*/
    else if (fabs(alpha_1) <= tol)
    {
        /**
		Reduce B to
		0 0 +
		0 x x
		0 0 x
		*/
        GivensRotation r1(0, 1);
        r1.computeUnconventional(B[0][1], B[1][1]);
        r1.rowRotation3(B);
        r1.columnRotation3(U);

        /**
		Reduce B to
		0 0 0
		0 x x
		0 + x
		*/
        GivensRotation r2(0, 2);
        r2.computeUnconventional(B[0][2], B[2][2]);
        r2.rowRotation3(B);
        r2.columnRotation3(U);

        process(1, B, U, sigma, V);
        sort1(U, sigma, V);
    }

    return count;
}
