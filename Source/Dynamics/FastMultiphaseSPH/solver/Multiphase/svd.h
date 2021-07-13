#pragma once
#include "../../math/geometry.h"

struct cmat2
{
    float  data[4];
    HDFUNC cmat2()
    {
        for (int i = 0; i < 4; i++)
            data[i] = 0;
    }
    HDFUNC cmat2(float a00, float a01, float a10, float a11)
    {
        data[0] = a00;
        data[1] = a01;
        data[2] = a10;
        data[3] = a11;
    }
    HDFUNC cmat2(float mat[])
    {  //make sure mat longer than 4
        for (int i = 0; i < 4; i++)
            data[i] = mat[i];
    }

    HDFUNC void Set(float mat[])
    {
        for (int i = 0; i < 4; i++)
            data[i] = mat[i];
    }
    HDFUNC void Set(float c)
    {
        for (int i = 0; i < 4; i++)
            data[i] = c;
    }
    HDFUNC void Print()
    {
        for (int i = 0; i < 2; i++)
        {
            printf("%f %f\n", data[i * 2], data[i * 2 + 1]);
        }
    }
    HDFUNC float* operator[](int i)
    {
        return &data[i * 2];
    }
    HDFUNC cmat3 operator+(cmat3& b)
    {
        cmat3 res;
        for (int k = 0; k < 4; k++)
        {
            res.data[k] = data[k] + b.data[k];
        }
        return res;
    }
    HDFUNC cmat3 operator*(float b)
    {
        cmat3 res;
        for (int k = 0; k < 4; k++)
        {
            res.data[k] = data[k] * b;
        }
        return res;
    }

    HDFUNC float Det()
    {
        return data[0] * data[3] - data[1] * data[2];
    }
    HDFUNC cmat2 Inv()
    {
        cmat2 inv;
        float det = Det();
        if (fabs(det) < 1e-10)
        {
            return inv;
        }
        inv[0][0] = (*this)[1][1] / det;
        inv[0][1] = -(*this)[0][1] / det;
        inv[1][0] = -(*this)[1][0] / det;
        inv[1][1] = (*this)[0][0] / det;
        return inv;
    }

    HDFUNC float Norm()
    {
        float norm = 0;
        for (int k = 0; k < 4; k++)
            norm += data[k] * data[k];
        return sqrt(norm);
    }

    HDFUNC void Add(cmat3& b)
    {
        for (int k = 0; k < 4; k++)
            data[k] = data[k] + b.data[k];
    }
    HDFUNC void Multiply(float b)
    {
        for (int k = 0; k < 4; k++)
            data[k] = data[k] * b;
    }
    HDFUNC cmat2 transpose()
    {
        cmat2 ret;
        ret[0][0] = (*this)[0][0];
        ret[0][1] = (*this)[1][0];
        ret[1][0] = (*this)[0][1];
        ret[1][1] = (*this)[1][1];
        return ret;
    }
};

class GivensRotation
{
public:
    int   rowi;
    int   rowk;
    float c;
    float s;

    HDFUNC inline GivensRotation(int rowi_in, int rowk_in)
        : rowi(rowi_in), rowk(rowk_in), c(1), s(0) {}
    HDFUNC inline GivensRotation(float a, float b, int rowi_in, int rowk_in)
        : rowi(rowi_in), rowk(rowk_in)
    {
        compute(a, b);
    }
    HDFUNC inline void transposeInPlace()
    {
        s = -s;
    }
    HDFUNC inline void compute(const float a, const float b)
    {
        float d = a * a + b * b;
        c       = 1;
        s       = 0;
        if (!ZERO(d))
        {
            float t = 1 / sqrt(d);
            c       = a * t;
            s       = -b * t;
        }
    }
    HDFUNC inline void computeUnconventional(const float a, const float b)
    {
        float d = a * a + b * b;
        c       = 0;
        s       = 1;
        if (d != 0)
        {
            float t = 1 / sqrt(d);
            s       = a * t;
            c       = b * t;
        }
    }
    HDFUNC inline void fill2(cmat2& R) const
    {
        R[rowi][rowi] = c;
        R[rowk][rowi] = -s;
        R[rowi][rowk] = s;
        R[rowk][rowk] = c;
    }
    HDFUNC inline void fill3(cmat3& R) const
    {
        make_identity(R);
        R[rowi][rowi] = c;
        R[rowk][rowi] = -s;
        R[rowi][rowk] = s;
        R[rowk][rowk] = c;
    }

    HDFUNC inline void rowRotation2(cmat2& A) const
    {
        for (int j = 0; j < 2; j++)
        {
            float tau1 = A[rowi][j];
            float tau2 = A[rowk][j];
            A[rowi][j] = c * tau1 - s * tau2;
            A[rowk][j] = s * tau1 + c * tau2;
        }
    }

    HDFUNC inline void rowRotation3(cmat3& A) const
    {
        for (int j = 0; j < 3; j++)
        {
            float tau1 = A[rowi][j];
            float tau2 = A[rowk][j];
            A[rowi][j] = c * tau1 - s * tau2;
            A[rowk][j] = s * tau1 + c * tau2;
        }
    }

    HDFUNC inline void columnRotation2(cmat2& A) const
    {
        for (int j = 0; j < 2; j++)
        {
            float tau1 = A[j][rowi];
            float tau2 = A[j][rowk];
            A[j][rowi] = c * tau1 - s * tau2;
            A[j][rowk] = s * tau1 + c * tau2;
        }
    }
    HDFUNC inline void columnRotation3(cmat3& A) const
    {
        for (int j = 0; j < 3; j++)
        {
            float tau1 = A[j][rowi];
            float tau2 = A[j][rowk];
            A[j][rowi] = c * tau1 - s * tau2;
            A[j][rowk] = s * tau1 + c * tau2;
        }
    }

    HDFUNC inline void operator*=(const GivensRotation& A)
    {
        float new_c = c * A.c - s * A.s;
        float new_s = s * A.c + c * A.s;
        c           = new_c;
        s           = new_s;
    }

    HDFUNC inline GivensRotation operator*(const GivensRotation& A) const
    {
        GivensRotation r(*this);
        r *= A;
        return r;
    }
};

struct adjacent
{
    int up;
    int down;
    int left;
    int right;
    int upright;
    int downright;
    int downleft;
    int upleft;
    int upup;
    int downdown;
    int leftleft;
    int rightright;

    HDFUNC int& operator[](int index)
    {
        switch (index)
        {
            case 0:
                return up;
            case 1:
                return down;
            case 2:
                return left;
            case 3:
                return right;
            case 4:
                return upright;
            case 5:
                return downright;
            case 6:
                return downleft;
            case 7:
                return upleft;
            case 8:
                return upup;
            case 9:
                return downdown;
            case 10:
                return leftleft;
            case 11:
                return rightright;
        }
    }
};

HDFUNC int singularValueDecomposition(
    cmat3&   A,
    cmat3&   U,
    cfloat3& sigma,
    cmat3&   V,
    float    tol = 128 * 1e-8);

HDFUNC inline void svd(
    cmat3&   A,
    cmat3&   U,
    cfloat3& sigma,
    cmat3&   V,
    float    tol = 128 * 1e-8)
{
    singularValueDecomposition(A, U, sigma, V, tol);
}

HDFUNC inline cmat3 MooreInv(cmat3& A)
{
    cmat3   U, V;
    cfloat3 sigma;
    singularValueDecomposition(A, U, sigma, V);
    for (int i = 0; i < 3; i++)
    {
        if (ZERO(sigma[i]))
            sigma[i] = 0.0f;
        else
            sigma[i] = 1.0 / sigma[i];
    }
    cmat3 S;
    S[0][0] = sigma.x;
    S[1][1] = sigma.y;
    S[2][2] = sigma.z;
    cmat3 UT;
    mat3transpose(U, UT);
    mat3prod(V, S, U);   //V*S
    mat3prod(U, UT, V);  //V*S*UT
    return V;
}