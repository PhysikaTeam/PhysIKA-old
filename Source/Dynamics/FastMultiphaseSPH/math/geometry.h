#pragma once

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "../pbal/core/math/vec.h"

#include <host_defines.h>
#define HDFUNC __host__ __device__

#define EPSILON 1e-6f

typedef unsigned int  uint;
typedef unsigned char uchar;

#define ZERO(a) (fabsf(a) < 1e-8f)

HDFUNC inline float cmin(float a, float b)
{
    return a < b ? a : b;
}

inline float cmax(float a, float b)
{
    return a > b ? a : b;
}

typedef pbal::Vec3f cfloat3;
typedef pbal::Vec2f cfloat2;
typedef pbal::Vec3i cint3;
typedef pbal::Vec2i cint2;
typedef pbal::Vec4f cfloat4;

inline float dot(cfloat2& a, cfloat2& b)
{
    return a.x * b.x + a.y * b.y;
}

HDFUNC inline cfloat3 minfilter(cfloat3 a, cfloat3 b)
{
    return cfloat3(cmin(a.x, b.x), cmin(a.y, b.y), cmin(a.z, b.z));
}

inline cfloat3 maxfilter(cfloat3 a, cfloat3 b)
{
    return cfloat3(cmax(a.x, b.x), cmax(a.y, b.y), cmax(a.z, b.z));
}

HDFUNC inline cfloat3 cross(cfloat3 a, cfloat3 b)
{
    cfloat3 tmp;
    tmp.x = a.y * b.z - a.z * b.y;
    tmp.y = a.z * b.x - a.x * b.z;
    tmp.z = a.x * b.y - a.y * b.x;
    return tmp;
}

HDFUNC inline float dot(cfloat3& a, cfloat3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float angle(cfloat3& a, cfloat3& b);

inline HDFUNC float cos(cfloat3& a, cfloat3& b)
{
    return dot(a, b) / sqrt(a.square() * b.square());
}

inline HDFUNC int fmax(int a, int b)
{
    return a > b ? a : b;
}

HDFUNC inline cint3 floor(cfloat3 a)
{
    return cint3(floor(a.x), floor(a.y), floor(a.z));
}
HDFUNC inline cint3 ceil(cfloat3 a)
{
    return cint3(ceil(a.x), ceil(a.y), ceil(a.z));
}

namespace pbal {

template <typename T, int d>
class Matrix
{
public:
    int len_ = d * d;
    T   data[d * d];

    Matrix()
    {
        for (int i = 0; i < len_; i++)
            data[i] = 0;
    }
    void Set(T mat[])
    {
        for (int i = 0; i < len_; i++)
            data[i] = mat[i];
    }
    void Set(T c)
    {
        for (int i = 0; i < len_; i++)
            data[i] = c;
    }
    T* operator[](int i)
    {
        return &data[i * d];
    }

    HDFUNC Matrix operator+(const Matrix& b)
    {
        Matrix res;
        for (int k = 0; k < len_; k++)
        {
            res.data[k] = data[k] + b.data[k];
        }
        return res;
    }
    HDFUNC Matrix operator-(const Matrix& b)
    {
        Matrix res;
        for (int k = 0; k < len_; k++)
        {
            res.data[k] = data[k] - b.data[k];
        }
        return res;
    }
    HDFUNC Matrix operator*(float b)
    {
        Matrix res;
        for (int k = 0; k < len_; k++)
        {
            res.data[k] = data[k] * b;
        }
        return res;
    }
    HDFUNC void operator*=(float b)
    {
        for (int k = 0; k < len_; k++)
            data[k] *= b;
    }
    HDFUNC void operator+=(const Matrix& b)
    {
        for (int k = 0; k < len_; k++)
            data[k] += b.data[k];
    }

    HDFUNC Matrix Reci()
    {
        Matrix ret;
        for (int i = 0; i < len_; i++)
        {
            if (ZERO(data[i]))
                ret.data[i] = 0.0f;
            else
                ret.data[i] = 1 / data[i];
        }
        return ret;
    }

    HDFUNC float length()
    {
        float norm = 0;
        for (int k = 0; k < len_; k++)
            norm += data[k] * data[k];
        return sqrtf(norm);
    }

    HDFUNC void Add(Matrix& b)
    {
        for (int k = 0; k < len_; k++)
            data[k] += b.data[k];
    }
    HDFUNC void Minus(Matrix& b)
    {
        for (int k = 0; k < len_; k++)
            data[k] -= b.data[k];
    }
    HDFUNC void Multiply(float b)
    {
        for (int k = 0; k < len_; k++)
            data[k] *= b;
    }
    HDFUNC Matrix GetTranspose()
    {
        Matrix res;
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                res[i][j] = (*this)[j][i];
        return res;
    }
};

template <typename T>
class Matrix<T, 3>
{
public:
    int len_ = 3 * 3;
    T   data[3 * 3];

    HDFUNC Matrix()
    {
        for (int i = 0; i < len_; i++)
            data[i] = 0;
    }
    HDFUNC void Set(T mat[])
    {
        for (int i = 0; i < len_; i++)
            data[i] = mat[i];
    }
    HDFUNC void Set(T c)
    {
        for (int i = 0; i < len_; i++)
            data[i] = c;
    }
    HDFUNC T* operator[](int i)
    {
        return &data[i * 3];
    }

    HDFUNC Matrix operator+(const Matrix& b)
    {
        Matrix res;
        for (int k = 0; k < len_; k++)
        {
            res.data[k] = data[k] + b.data[k];
        }
        return res;
    }
    HDFUNC Matrix operator-(const Matrix& b)
    {
        Matrix res;
        for (int k = 0; k < len_; k++)
        {
            res.data[k] = data[k] - b.data[k];
        }
        return res;
    }
    HDFUNC Matrix operator*(float b)
    {
        Matrix res;
        for (int k = 0; k < len_; k++)
        {
            res.data[k] = data[k] * b;
        }
        return res;
    }
    HDFUNC void operator*=(float b)
    {
        for (int k = 0; k < len_; k++)
            data[k] *= b;
    }
    HDFUNC void operator+=(const Matrix& b)
    {
        for (int k = 0; k < len_; k++)
            data[k] += b.data[k];
    }

    HDFUNC Matrix Reci()
    {
        Matrix ret;
        for (int i = 0; i < len_; i++)
        {
            if (ZERO(data[i]))
                ret.data[i] = 0.0f;
            else
                ret.data[i] = 1 / data[i];
        }
        return ret;
    }

    HDFUNC float length()
    {
        float norm = 0;
        for (int k = 0; k < len_; k++)
            norm += data[k] * data[k];
        return sqrtf(norm);
    }

    HDFUNC void Add(Matrix& b)
    {
        for (int k = 0; k < len_; k++)
            data[k] += b.data[k];
    }
    HDFUNC void Minus(Matrix& b)
    {
        for (int k = 0; k < len_; k++)
            data[k] -= b.data[k];
    }
    HDFUNC void Multiply(float b)
    {
        for (int k = 0; k < len_; k++)
            data[k] *= b;
    }
    HDFUNC T Det()
    {
        return data[0] * (data[4] * data[8] - data[5] * data[7]) + data[1] * (data[5] * data[6] - data[3] * data[8]) + data[2] * (data[3] * data[7] - data[4] * data[6]);
    }
    HDFUNC Matrix Inv()
    {
        Matrix inv;
        float  det = Det();
        if (fabs(det) < 1e-10)
        {
            return inv;
        }
        inv[0][0] = (data[4] * data[8] - data[5] * data[7]) / det;
        inv[0][1] = (data[2] * data[7] - data[1] * data[8]) / det;
        inv[0][2] = (data[1] * data[5] - data[2] * data[4]) / det;
        inv[1][0] = (data[5] * data[6] - data[3] * data[8]) / det;
        inv[1][1] = (data[0] * data[8] - data[2] * data[6]) / det;
        inv[1][2] = (data[2] * data[3] - data[0] * data[5]) / det;
        inv[2][0] = (data[3] * data[7] - data[4] * data[6]) / det;
        inv[2][1] = (data[1] * data[6] - data[0] * data[7]) / det;
        inv[2][2] = (data[0] * data[4] - data[1] * data[3]) / det;
        return inv;
    }
    HDFUNC Matrix GetTranspose()
    {
        Matrix res;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                res[i][j] = (*this)[j][i];
        return res;
    }
    HDFUNC cfloat3 Col(int col)
    {
        return cfloat3((*this)[0][col], (*this)[1][col], (*this)[2][col]);
    }
};

}  // namespace pbal

using mat2f = pbal::Matrix<float, 2>;
using cmat3 = pbal::Matrix<float, 3>;

inline mat2f TensorProduct(cfloat2& a, cfloat2& b)
{
    mat2f res;
    res[0][0] = a.x * b.x;
    res[0][1] = a.x * b.y;
    res[1][0] = a.y * b.x;
    res[1][1] = a.y * b.y;
    return res;
}

HDFUNC __inline__ cmat3 TensorProduct(cfloat3& a, cfloat3& b)
{
    cmat3 res;
    res[0][0] = a.x * b.x;
    res[0][1] = a.x * b.y;
    res[0][2] = a.x * b.z;
    res[1][0] = a.y * b.x;
    res[1][1] = a.y * b.y;
    res[1][2] = a.y * b.z;
    res[2][0] = a.z * b.x;
    res[2][1] = a.z * b.y;
    res[2][2] = a.z * b.z;
    return res;
}

HDFUNC __inline__ void mat3add(cmat3& a, cmat3& b, cmat3& c)
{
    for (int i = 0; i < 9; i++)
        c.data[i] = a.data[i] + b.data[i];
}

HDFUNC __inline__ void mat3sub(cmat3& a, cmat3& b, cmat3& c)
{
    for (int i = 0; i < 9; i++)
        c.data[i] = a.data[i] - b.data[i];
}

HDFUNC __inline__ void mat3prod(cmat3& a, cmat3& b, cmat3& c)
{
    c[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0];
    c[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1];
    c[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2];

    c[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0];
    c[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1];
    c[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2];

    c[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0];
    c[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1];
    c[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2];
}

HDFUNC __inline__ void mat3transpose(cmat3& a, cmat3& b)
{
    cmat3& tmp = b;
    tmp[0][0]  = a[0][0];
    tmp[0][1]  = a[1][0];
    tmp[0][2]  = a[2][0];
    tmp[1][0]  = a[0][1];
    tmp[1][1]  = a[1][1];
    tmp[1][2]  = a[2][1];
    tmp[2][0]  = a[0][2];
    tmp[2][1]  = a[1][2];
    tmp[2][2]  = a[2][2];
}

HDFUNC inline void make_identity(cmat3& mat)
{
    for (int i = 0; i < 9; i++)
        mat.data[i] = 0;
    mat[0][0] = mat[1][1] = mat[2][2] = 1;
}

/*
Compute the product of cmat3 m and cfloat3 v,
storing the result into cfloat3 c.
c and v can be the same vector.
*/

inline void mvprod(mat2f& m, cfloat2& v, cfloat2& c)
{
    cfloat2 tmp;
    tmp.x = m[0][0] * v.x + m[0][1] * v.y;
    tmp.y = m[1][0] * v.x + m[1][1] * v.y;
    c     = tmp;
}

inline void MatProduct(mat2f& a, mat2f& b, mat2f& c)
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
        {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
}

HDFUNC __inline__ void MatProduct(cmat3& a, cmat3& b, cmat3& c)
{
    c[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0];
    c[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1];
    c[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2];

    c[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0];
    c[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1];
    c[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2];

    c[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0];
    c[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1];
    c[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2];
}

HDFUNC __inline__ void mvprod(cmat3& m, cfloat3& v, cfloat3& c)
{
    cfloat3 tmp;
    tmp.x = m.data[0] * v.x + m.data[1] * v.y + m.data[2] * v.z;
    tmp.y = m.data[3] * v.x + m.data[4] * v.y + m.data[5] * v.z;
    tmp.z = m.data[6] * v.x + m.data[7] * v.y + m.data[8] * v.z;
    c     = tmp;
}

void RotateX(cfloat3& a, float b);
void RotateY(cfloat3& a, float b);
void RotateZ(cfloat3& a, float b);
void RotateXYZ(cfloat3& a, cfloat3& xyz);

HDFUNC __inline__ void AxisAngle2Matrix(cfloat3 axis, float angle, cmat3& mat)
{
    //Do the normalization if needed.
    float l = axis.length();
    if (fabs(l - 1) > EPSILON)
        axis /= l;
    float s   = sin(angle);
    float c   = cos(angle);
    float t   = 1 - c;
    mat[0][0] = t * axis.x * axis.x + c;
    mat[1][1] = t * axis.y * axis.y + c;
    mat[2][2] = t * axis.z * axis.z + c;

    float tmp1 = axis.x * axis.y * t;
    float tmp2 = axis.z * s;
    mat[1][0]  = tmp1 + tmp2;
    mat[0][1]  = tmp1 - tmp2;
    tmp1       = axis.x * axis.z * t;
    tmp2       = axis.y * s;
    mat[2][0]  = tmp1 - tmp2;
    mat[0][2]  = tmp1 + tmp2;
    tmp1       = axis.y * axis.z * t;
    tmp2       = axis.x * s;
    mat[2][1]  = tmp1 + tmp2;
    mat[1][2]  = tmp1 - tmp2;
}

/* -----------------------------------


                OpenGL Utility


-------------------------------------*/

const float cPI = 3.1415926536;

struct cvertex
{
    float position[4];
    float color[4];
};

//watch out, column major
struct cmat4
{
    float  data[16];
    float* operator[](int i)
    {
        return &data[i * 4];
    }
    cmat4 operator*(cmat4& b)
    {
        cmat4 tmp;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                tmp[i][j] = data[0 + i * 4] * b[0][j] + data[1 + i * 4] * b[1][j]
                            + data[2 + i * 4] * b[2][j] + data[3 + i * 4] * b[3][j];
            }
        }
        return tmp;
    }

    cfloat4 operator*(cfloat4& b)
    {
        cfloat4 tmp;
        tmp.x = data[0 * 4 + 0] * b.x + data[1 * 4 + 0] * b.y + data[2 * 4 + 0] * b.z + data[3 * 4 + 0] * b.w;
        tmp.y = data[0 * 4 + 1] * b.x + data[1 * 4 + 1] * b.y + data[2 * 4 + 1] * b.z + data[3 * 4 + 1] * b.w;
        tmp.z = data[0 * 4 + 2] * b.x + data[1 * 4 + 2] * b.y + data[2 * 4 + 2] * b.z + data[3 * 4 + 2] * b.w;
        tmp.w = data[0 * 4 + 3] * b.x + data[1 * 4 + 3] * b.y + data[2 * 4 + 3] * b.z + data[3 * 4 + 3] * b.w;
        return tmp;
    }
};

extern const cmat4 IDENTITY_MAT;

float cotangent(float angle);
float deg2rad(float deg);
float rad2deg(float rad);

void RotateAboutX(cmat4& m, float ang);
void RotateAboutY(cmat4& m, float ang);
void RotateAboutZ(cmat4& m, float ang);
void ScaleMatrix(cmat4& m, cfloat3 x);
void TranslateMatrix(cmat4& m, cfloat3 x);

cmat4 CreateProjectionMatrix(float fovy, float aspect_ratio, float near_plane, float far_plane);

struct vertex
{
    cfloat3 pos;
    cfloat4 color;
};