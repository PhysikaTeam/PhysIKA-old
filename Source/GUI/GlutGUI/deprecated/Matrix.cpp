#include "Matrix.h"
namespace gui {
template <typename T>
void Transform3D<T>::Translate(const Vector3f& trans)
{
    Identity();

    x[12] = trans.x;
    x[13] = trans.y;
    x[14] = trans.z;
}

template <typename T>
void Transform3D<T>::TranslateX(const float& dist)
{
    Identity();
    x[12] = dist;
}

template <typename T>
void Transform3D<T>::TranslateY(const float& dist)
{
    Identity();
    x[13] = dist;
}

template <typename T>
void Transform3D<T>::TranslateZ(const float& dist)
{
    Identity();
    x[14] = dist;
}

template <typename T>
void Transform3D<T>::Rotate(const float angle, Vector3f& axis)
{
    double a = DEGTORAD(angle);
    float  c = cos(a);
    float  s = sin(a);
    axis.Normalize();

    float ux = axis.x;
    float uy = axis.y;
    float uz = axis.z;

    float minc = 1 - c;

    x[0] = c + ( minc )*pow(ux, 2);
    x[1] = ( minc )*ux * uy + s * uz;
    x[2] = ( minc )*ux * uz - s * uy;
    x[3] = 0;

    x[4] = ( minc )*uy * ux - s * uz;
    x[5] = c + ( minc )*pow(uy, 2);
    x[6] = ( minc )*uy * uz + s * ux;
    x[7] = 0;

    x[8]  = ( minc )*uz * ux + s * uy;
    x[9]  = ( minc )*uy * uz - s * ux;
    x[10] = c + ( minc )*pow(uz, 2);
    x[11] = 0;

    x[12] = 0;
    x[13] = 0;
    x[14] = 0;
    x[15] = 1;
}

template <typename T>
void Transform3D<T>::RotateX(const float& angle)
{
    float s = sin(DEGTORAD(angle));
    float c = cos(DEGTORAD(angle));

    Identity();

    x[5]  = c;
    x[6]  = s;
    x[9]  = -s;
    x[10] = c;
}

template <typename T>
void Transform3D<T>::RotateY(const float& angle)
{
    float s = sin(DEGTORAD(angle));
    float c = cos(DEGTORAD(angle));

    Identity();

    x[0]  = c;
    x[2]  = -s;
    x[8]  = s;
    x[10] = c;
}

template <typename T>
void Transform3D<T>::RotateZ(const float& angle)
{
    float s = sin(DEGTORAD(angle));
    float c = cos(DEGTORAD(angle));

    identity();

    x[0] = c;
    x[1] = s;
    x[4] = -s;
    x[5] = c;
}

template <typename T>
Transform3D<T> Transform3D<T>::Invert()
{
    Transform3D result;

#define m11 (*this)(0, 0)
#define m12 (*this)(0, 1)
#define m13 (*this)(0, 2)
#define m14 (*this)(0, 3)
#define m21 (*this)(1, 0)
#define m22 (*this)(1, 1)
#define m23 (*this)(1, 2)
#define m24 (*this)(1, 3)
#define m31 (*this)(2, 0)
#define m32 (*this)(2, 1)
#define m33 (*this)(2, 2)
#define m34 (*this)(2, 3)
#define m41 (*this)(3, 0)
#define m42 (*this)(3, 1)
#define m43 (*this)(3, 2)
#define m44 (*this)(3, 3)

    // Inverse = adjoint / det. (See linear algebra texts.)

    // pre-compute 2x2 dets for last two rows when computing
    // cofactors of first two rows.
    T d12 = (m31 * m42 - m41 * m32);
    T d13 = (m31 * m43 - m41 * m33);
    T d23 = (m32 * m43 - m42 * m33);
    T d24 = (m32 * m44 - m42 * m34);
    T d34 = (m33 * m44 - m43 * m34);
    T d41 = (m34 * m41 - m44 * m31);

    T tmp[16];

    tmp[0] = (m22 * d34 - m23 * d24 + m24 * d23);
    tmp[1] = -(m21 * d34 + m23 * d41 + m24 * d13);
    tmp[2] = (m21 * d24 + m22 * d41 + m24 * d12);
    tmp[3] = -(m21 * d23 - m22 * d13 + m23 * d12);

    // Compute determinant as early as possible using these cofactors.
    T det = m11 * tmp[0] + m12 * tmp[1] + m13 * tmp[2] + m14 * tmp[3];

    // Run singularity test.
    if (det == 0.0)
    {

        T identity[16] = {
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0
        };

        memcpy(result.m, identity, 16 * sizeof(T));
    }
    else
    {
        T invDet = 1.0f / det;

        // Compute rest of inverse.
        tmp[0] *= invDet;
        tmp[1] *= invDet;
        tmp[2] *= invDet;
        tmp[3] *= invDet;

        tmp[4] = -(m12 * d34 - m13 * d24 + m14 * d23) * invDet;
        tmp[5] = (m11 * d34 + m13 * d41 + m14 * d13) * invDet;
        tmp[6] = -(m11 * d24 + m12 * d41 + m14 * d12) * invDet;
        tmp[7] = (m11 * d23 - m12 * d13 + m13 * d12) * invDet;

        // Pre-compute 2x2 dets for first two rows when computing cofactors
        // of last two rows.
        d12 = m11 * m22 - m21 * m12;
        d13 = m11 * m23 - m21 * m13;
        d23 = m12 * m23 - m22 * m13;
        d24 = m12 * m24 - m22 * m14;
        d34 = m13 * m24 - m23 * m14;
        d41 = m14 * m21 - m24 * m11;

        tmp[8]  = (m42 * d34 - m43 * d24 + m44 * d23) * invDet;
        tmp[9]  = -(m41 * d34 + m43 * d41 + m44 * d13) * invDet;
        tmp[10] = (m41 * d24 + m42 * d41 + m44 * d12) * invDet;
        tmp[11] = -(m41 * d23 - m42 * d13 + m43 * d12) * invDet;
        tmp[12] = -(m32 * d34 - m33 * d24 + m34 * d23) * invDet;
        tmp[13] = (m31 * d34 + m33 * d41 + m34 * d13) * invDet;
        tmp[14] = -(m31 * d24 + m32 * d41 + m34 * d12) * invDet;
        tmp[15] = (m31 * d23 - m32 * d13 + m33 * d12) * invDet;

        memcpy(result.x, tmp, 16 * sizeof(T));
    }

#undef m11
#undef m12
#undef m13
#undef m14
#undef m21
#undef m22
#undef m23
#undef m24
#undef m31
#undef m32
#undef m33
#undef m34
#undef m41
#undef m42
#undef m43
#undef m44

    return result;
}

}  // namespace gui