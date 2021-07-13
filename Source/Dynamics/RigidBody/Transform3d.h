#pragma once

#include "Core/Matrix/matrix_mxn.h"
#include "Core/Vector/vector_nd.h"
#include "Core/Matrix/matrix_3x3.h"
#include "Core/Quaternion/quaternion.h"
#include "Inertia.h"

#include <memory>

namespace PhysIKA {
/**
    * @brief Transformation for spatial vector
    * 
    * @note
    * A, B are frames with origins at O and P.
    * That is: r is Op vector in A frame. q rotate a vector from A frame to B frame.
    */

template <typename T>
class Transform3d
{
public:
    Transform3d();

    Transform3d(const VectorBase<float>& r, const Quaternion<float>& q);
    Transform3d(const Vector3f& r, const Quaternion<float>& q);
    Transform3d(const Vector3f& r, const Matrix3f& m);

    void set(const Vector3f& r, const Quaternion<float>& q);
    void setTranslation(const VectorBase<float>& r);
    void setRotation(const Quaternion<float>& q);

    const SpatialVector<T>& transformF(const SpatialVector<T>& f) const;

    //MatrixMN<T> transformF(const MatrixMN<T>& f);

    const SpatialVector<T>& transformM(const SpatialVector<T>& m) const;

    const Inertia<T>  transformI(const Inertia<T>& inertia) const;
    const MatrixMN<T> transformI(const MatrixMN<T>& inertia) const;

    //MatrixMN<T> transformM(const MatrixMN<T>& m);

    // Merge two transformation into one
    // X = X1 * X2
    const Transform3d<T> operator*(const Transform3d<T>& trans) const;

    const Transform3d<T> inverseTransform() const;

    const Matrix3f& getRotationMatrix() const
    {
        return m_rotation;
    }
    const Quaternion<float>& getRotation() const
    {
        return m_rotation_q;
    }
    const Vector3f& getTranslation() const
    {
        return m_translation;
    }

private:
    Quaternion<float> m_rotation_q;
    Matrix3f          m_rotation;
    Vector3f          m_translation;
};

template <typename T>
inline Transform3d<T>::Transform3d()
    : m_translation(0), m_rotation(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
{
}
template <typename T>
inline Transform3d<T>::Transform3d(const VectorBase<float>& r, const Quaternion<float>& q)
{
    m_translation[0] = r[0];
    m_translation[1] = r[1];
    m_translation[2] = r[2];

    m_rotation   = q.get3x3Matrix();
    m_rotation_q = q;
}
template <typename T>
inline Transform3d<T>::Transform3d(const Vector3f& r, const Quaternion<float>& q)
    : m_translation(r)
{
    m_rotation   = q.get3x3Matrix();
    m_rotation_q = q;
}
template <typename T>
inline Transform3d<T>::Transform3d(const Vector3f& r, const Matrix3f& m)
    : m_translation(r), m_rotation(m), m_rotation_q(m)
{
}
template <typename T>
inline void Transform3d<T>::set(const Vector3f& r, const Quaternion<float>& q)
{
    //m_translation = r;
    m_translation[0] = r[0];
    m_translation[1] = r[1];
    m_translation[2] = r[2];

    m_rotation   = q.get3x3Matrix();
    m_rotation_q = q;
}
template <typename T>
inline void Transform3d<T>::setTranslation(const VectorBase<float>& r)
{
    m_translation[0] = r[0];
    m_translation[1] = r[1];
    m_translation[2] = r[2];
}
template <typename T>
inline void Transform3d<T>::setRotation(const Quaternion<float>& q)
{
    m_rotation   = q.get3x3Matrix();
    m_rotation_q = q;
}
template <typename T>
inline const SpatialVector<T>& Transform3d<T>::transformF(const SpatialVector<T>& f) const
{
    SpatialVector<T> res;
    // translation:  new_torque = torque - r x f;
    res[0] = f[0];
    res[1] = f[1];
    res[2] = f[2];
    res[0] -= m_translation[1] * f[5] - m_translation[2] * f[4];
    res[1] -= m_translation[2] * f[3] - m_translation[0] * f[5];
    res[2] -= m_translation[0] * f[4] - m_translation[1] * f[3];

    // translation: new_f = f;
    res[3] = f[3];
    res[4] = f[4];
    res[5] = f[5];

    // rotation: new_torque = rotate * torque;
    T tmpx, tmpy, tmpz;
    tmpx   = m_rotation(0, 0) * res[0] + m_rotation(0, 1) * res[1] + m_rotation(0, 2) * res[2];
    tmpy   = m_rotation(1, 0) * res[0] + m_rotation(1, 1) * res[1] + m_rotation(1, 2) * res[2];
    tmpz   = m_rotation(2, 0) * res[0] + m_rotation(2, 1) * res[1] + m_rotation(2, 2) * res[2];
    res[0] = tmpx;
    res[1] = tmpy;
    res[2] = tmpz;

    // rotation: new_f = rotate * f;
    tmpx   = m_rotation(0, 0) * res[3] + m_rotation(0, 1) * res[4] + m_rotation(0, 2) * res[5];
    tmpy   = m_rotation(1, 0) * res[3] + m_rotation(1, 1) * res[4] + m_rotation(1, 2) * res[5];
    tmpz   = m_rotation(2, 0) * res[3] + m_rotation(2, 1) * res[4] + m_rotation(2, 2) * res[5];
    res[3] = tmpx;
    res[4] = tmpy;
    res[5] = tmpz;

    return res;
}
template <typename T>
inline const SpatialVector<T>& Transform3d<T>::transformM(const SpatialVector<T>& m) const
{
    SpatialVector<T> res;
    // translation: new_w = w;
    res[0] = m[0];
    res[1] = m[1];
    res[2] = m[2];

    // translation:  new_v = v - r x w;
    res[3] = m[3];
    res[4] = m[4];
    res[5] = m[5];
    res[3] -= m_translation[1] * m[2] - m_translation[2] * m[1];
    res[4] -= m_translation[2] * m[0] - m_translation[0] * m[2];
    res[5] -= m_translation[0] * m[1] - m_translation[1] * m[0];

    // rotation: new_w = rotate * w;
    T tmpx, tmpy, tmpz;
    tmpx   = m_rotation(0, 0) * res[0] + m_rotation(0, 1) * res[1] + m_rotation(0, 2) * res[2];
    tmpy   = m_rotation(1, 0) * res[0] + m_rotation(1, 1) * res[1] + m_rotation(1, 2) * res[2];
    tmpz   = m_rotation(2, 0) * res[0] + m_rotation(2, 1) * res[1] + m_rotation(2, 2) * res[2];
    res[0] = tmpx;
    res[1] = tmpy;
    res[2] = tmpz;

    // rotation: new_v = rotate * v;
    tmpx   = m_rotation(0, 0) * res[3] + m_rotation(0, 1) * res[4] + m_rotation(0, 2) * res[5];
    tmpy   = m_rotation(1, 0) * res[3] + m_rotation(1, 1) * res[4] + m_rotation(1, 2) * res[5];
    tmpz   = m_rotation(2, 0) * res[3] + m_rotation(2, 1) * res[4] + m_rotation(2, 2) * res[5];
    res[3] = tmpx;
    res[4] = tmpy;
    res[5] = tmpz;

    return res;
}

// Transformation of inertia matrix
// I_2 = X_12f * I_1 * X_21m
template <typename T>
inline const MatrixMN<T> Transform3d<T>::transformI(const MatrixMN<T>& inertia) const
{
    MatrixMN<T> res(6, 6);

    /// I_tmp = (X_12f * I_1)
    for (int i = 0; i < 6; ++i)
    {
        SpatialVector<T> tmpv(inertia(0, i), inertia(1, i), inertia(2, i), inertia(3, i), inertia(4, i), inertia(5, i));
        SpatialVector<T> tmpres = this->transformF(tmpv);

        /// res = (X_12f * I_1)  = I_tmp
        res(0, i) = tmpres[0];
        res(1, i) = tmpres[1];
        res(2, i) = tmpres[2];
        res(3, i) = tmpres[3];
        res(4, i) = tmpres[4];
        res(5, i) = tmpres[5];
    }

    /// I_2 = I_tmp * X_21m = (X_12f * I_tmp^T)^T
    for (int i = 0; i < 6; ++i)
    {
        SpatialVector<T> tmpv(res(i, 0), res(i, 1), res(i, 2), res(i, 3), res(i, 4), res(i, 5));
        SpatialVector<T> tmpres = this->transformF(tmpv);

        res(i, 0) = tmpres[0];
        res(i, 1) = tmpres[1];
        res(i, 2) = tmpres[2];
        res(i, 3) = tmpres[3];
        res(i, 4) = tmpres[4];
        res(i, 5) = tmpres[5];
    }

    return res;
}

// Merge tow transformation into one
// X1 = [ E1      0 ]       X2 = [ E2      0 ]
//        [-E1*r1x  E1]            [-E2*r2x  E2]
//==>
//X = X1 * X2 = [ E1*E2                  0    ] = [ E1*E2                     0    ]
//                [-E1*r1x*E2-E1*E2*r2x    E1*E2]   [-E1*E2*(E2^T*r1x*E2+r2x)   E1*E2]
//==>
// X = X1*X2 = [ E1*E2                        0    ]
//             [-E1*E2*(r2x + (E2^T*r1)x))    E1*E2]
//
// ==>
// E = E1*E2
// r = E2^T * r1 + r2
template <typename T>
inline const Transform3d<T> Transform3d<T>::operator*(const Transform3d<T>& trans) const
{
    Transform3d<T> res;
    res.m_rotation_q  = this->m_rotation_q * trans.m_rotation_q;
    res.m_rotation    = this->m_rotation * trans.m_rotation;
    res.m_translation = trans.m_translation + trans.m_rotation.transpose() * this->m_translation;
    return res;
}
template <typename T>
inline const Transform3d<T> Transform3d<T>::inverseTransform() const
{
    Transform3d<T> res(-((this->m_rotation) * (this->m_translation)), this->m_rotation.transpose());
    return res;
}
}  // namespace PhysIKA