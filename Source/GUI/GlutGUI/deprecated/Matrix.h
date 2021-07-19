#pragma once
#include "Vec.h"
#include <assert.h>

namespace gui {
/*!
 *    \struct    Matrix
 *    \brief    Template for 2D or 3D vector with components of any number type.
 */
template <int dim_m, int dim_n, typename T>
class Matrix
{
public:
    enum PARAMETER
    {
        m    = dim_m,
        n    = dim_n,
        size = dim_m * dim_n
    };
    T x[m * n + (m * n == 0)];

    Matrix()
    {
        for (int i = 0; i < size; i++)
            x[i] = ( T )0.0;
    }
    Matrix(T a)
    {
        for (int i = 0; i < size; i++)
            x[i] = a;
    }
    Matrix(const Matrix& A)
    {
        for (int i = 0; i < size; i++)
            x[i] = A.x[i];
    }

    int Rows() const
    {
        return m;
    }

    int Columns() const
    {
        return n;
    }

    T operator()(const int i, const int j) const
    {
        return x[i * n + j];
    }

    T& operator()(const int i, const int j)
    {
        return x[i * n + j];
    }

    void operator=(const Matrix& A)
    {
        assert(m == A.Rows() && n == A.Columns());
        for (int i = 0; i < size; i++)
            x[i] = A.x[i];
    }

    void operator+=(const Matrix& A)
    {
        for (int i = 0; i < size; i++)
            x[i] += A.x[i];
    }

    void operator-=(const Matrix& A)
    {
        for (int i = 0; i < size; i++)
            x[i] -= A.x[i];
    }

    void operator*=(const T a)
    {
        for (int i = 0; i < size; i++)
            x[i] *= a;
    }

    void operator/=(const T a)
    {
        for (int i = 0; i < size; i++)
            x[i] /= a;
    }

    Matrix operator+(const Matrix& A)
    {
        Matrix mat;
        for (int i = 0; i < size; i++)
            mat.x[i] = x[i] + A.x[i];
        return mat;
    }

    Matrix operator-(const Matrix& A)
    {
        Matrix mat;
        for (int i = 0; i < size; i++)
            mat.x[i] = x[i] - A.x[i];
        return mat;
    }

    template <int dim_t>
    Matrix<dim_m, dim_t, T> operator*(const Matrix<dim_n, dim_t, T>& A)
    {
        Matrix<m, A.n, T> mat;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < A.n; j++)
                for (int k = 0; k < n; k++)
                    mat(i, j) += (*this)(i, k) * A(k, j);
        return mat;
    }

    Vec<dim_m, T> operator*(const Vec<dim_n, T>& v)
    {
        Vec<m, T> result;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                result[i] += (*this)(i, j) * v[j];
        return result;
    }

    Matrix operator*(const T s)
    {
        Matrix mat;
        for (int i = 0; i < size; i++)
            mat.x[i] = x[i] * s;
        return mat;
    }
};

template <int dim, typename T>
class MatrixSq : public Matrix<dim, dim, T>
{
public:
    STATIC_ASSERT(m > 4);

    MatrixSq()
        : Matrix(){};
    MatrixSq(T a)
        : Matrix(a){};
    MatrixSq(const MatrixSq& A)
    {
        for (int i = 0; i < size; i++)
            x[i] = A.x[i];
    }
    MatrixSq(const Matrix<dim, dim, T>& A)
    {
        for (int i = 0; i < size; i++)
            x[i] = A.x[i];
    }

    MatrixSq Transpose()
    {
        Matrix mat;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                mat(i, j) = (*this)(j, i);
            }
        return mat;
    }

    static MatrixSq Identity()
    {
        MatrixSq mat;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                    mat(i, j) = 1;
                else
                    mat(i, j) = 0;
            }
        return mat;
    }
};

template <typename T>
class MatrixSq<4, T> : public Matrix<4, 4, T>
{
public:
    MatrixSq()
        : Matrix(){};
    MatrixSq(T a)
        : Matrix(a){};
    MatrixSq(const MatrixSq& A)
    {
        for (int i = 0; i < size; i++)
            x[i] = A.x[i];
    }
    MatrixSq(const Matrix<4, 4, T>& A)
    {
        for (int i = 0; i < size; i++)
            x[i] = A.x[i];
    }

    MatrixSq(float m0, float m4, float m8, float m12, float m1, float m5, float m9, float m13, float m2, float m6, float m10, float m14, float m3, float m7, float m11, float m15)
    {
        x[0]  = m0;
        x[4]  = m4;
        x[8]  = m8;
        x[12] = m12;
        x[1]  = m1;
        x[5]  = m5;
        x[9]  = m9;
        x[13] = m13;
        x[2]  = m2;
        x[6]  = m6;
        x[10] = m10;
        x[14] = m14;
        x[3]  = m3;
        x[7]  = m7;
        x[11] = m11;
        x[15] = m15;
    }

    MatrixSq Transpose()
    {
        MatrixSq<4, T> mat;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                mat(i, j) = (*this)(j, i);
            }
        return mat;
    }

    static MatrixSq Identity()
    {
        MatrixSq<4, T> mat;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                    mat(i, j) = 1;
                else
                    mat(i, j) = 0;
            }
        return mat;
    }
};

template <typename T>
class MatrixSq<3, T> : public Matrix<3, 3, T>
{
public:
    MatrixSq()
        : Matrix(){};
    MatrixSq(T a)
        : Matrix(a){};
    MatrixSq(const MatrixSq& A)
    {
        for (int i = 0; i < size; i++)
            x[i] = A.x[i];
    }
    MatrixSq(const Matrix<3, 3, T>& A)
    {
        for (int i = 0; i < size; i++)
            x[i] = A.x[i];
    }

    MatrixSq(float m0, float m1, float m2, float m3, float m4, float m5, float m6, float m7, float m8)
    {
        x[0] = m0;
        x[1] = m1;
        x[2] = m2;
        x[3] = m3;
        x[4] = m4;
        x[5] = m5;
        x[6] = m6;
        x[7] = m7;
        x[8] = m8;
    }

    MatrixSq Transpose()
    {
        MatrixSq<3, T> mat;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                mat(i, j) = (*this)(j, i);
            }
        return mat;
    }

    static MatrixSq Identity()
    {
        MatrixSq<3, T> mat;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                    mat(i, j) = 1;
                else
                    mat(i, j) = 0;
            }
        return mat;
    }

    T Norm2()
    {
        T result = 0.0;
        for (int i = 0; i < size; i++)
            result += x[i] * x[i];
        return sqrt(result);
    }
};

template <typename T>
class Rotation3D : public MatrixSq<3, T>
{
public:
    Rotation3D()
        : MatrixSq(){};
    Rotation3D(T a)
        : MatrixSq(a){};
    Rotation3D(const Matrix<3, 3, T>& A)
    {
        for (int i = 0; i < size; i++)
            x[i] = A.x[i];
    }
    Rotation3D(const Rotation3D& A)
    {
        for (int i = 0; i < size; i++)
            x[i] = A.x[i];
    }

    Rotation3D(float m0, float m3, float m6, float m1, float m4, float m7, float m2, float m5, float m8)
        : MatrixSq(m0, m3, m6, m1, m4, m7, m2, m5, m8)
    {
    }
};

template <typename T>
class Transform3D : public MatrixSq<4, T>
{
public:
    Transform3D()
        : MatrixSq(){};
    Transform3D(T a)
        : MatrixSq(a){};
    Transform3D(const Matrix<4, 4, T>& A)
    {
        for (int i = 0; i < size; i++)
            x[i] = A.x[i];
    }
    Transform3D(const Transform3D& A)
    {
        for (int i = 0; i < size; i++)
            x[i] = A.x[i];
    }

    Transform3D(float m0, float m4, float m8, float m12, float m1, float m5, float m9, float m13, float m2, float m6, float m10, float m14, float m3, float m7, float m11, float m15)
        : MatrixSq(m0, m4, m8, m12, m1, m5, m9, m13, m2, m6, m10, m14, m3, m7, m11, m15)
    {
    }

    void Translate(const Vector3f& trans);
    void TranslateX(const float& dist);
    void TranslateY(const float& dist);
    void TranslateZ(const float& dist);
    void Rotate(const float angle, Vector3f& axis);
    void RotateX(const float& angle);
    void RotateY(const float& angle);
    void RotateZ(const float& angle);

    Transform3D Invert();
};

template class MatrixSq<3, float>;
template class MatrixSq<3, double>;
template class MatrixSq<4, float>;
template class MatrixSq<4, double>;
template class Rotation3D<float>;
template class Rotation3D<double>;

typedef MatrixSq<3, float>  MatrixSq3f;
typedef MatrixSq<3, double> MatrixSq3d;
typedef MatrixSq<4, float>  MatrixSq4f;
typedef MatrixSq<4, double> MatrixSq4d;
typedef Rotation3D<float>   Rotation3Df;
typedef Rotation3D<double>  Rotation3Dd;

}  // namespace gui
