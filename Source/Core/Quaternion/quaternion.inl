/*
 * @file quaternion.cpp 
 * @brief quaternion.
 * @author Sheng Yang, Fei Zhu
 * 
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <cmath>
#include <cstdlib>
#include <iostream>
#include "Core/Utility.h"
#include "../Vector.h"
#include "../Matrix.h"
#include "quaternion.h"

namespace PhysIKA {

template <typename Real>
COMM_FUNC Quaternion<Real>::Quaternion()
    : x_(0), y_(0), z_(0), w_(1)
{
}

template <typename Real>
COMM_FUNC Quaternion<Real>::Quaternion(Real x, Real y, Real z, Real w)
    : x_(x), y_(y), z_(z), w_(w)
{
}

template <typename Real>
COMM_FUNC Quaternion<Real>::Quaternion(const Vector<Real, 3>& unit_axis, Real angle_rad)
{
    setRotation(unit_axis, angle_rad);
}

template <typename Real>
COMM_FUNC Quaternion<Real>::Quaternion(Real angle_rad, const Vector<Real, 3>& unit_axis)
{
    setRotation(unit_axis, angle_rad);
}

template <typename Real>
COMM_FUNC Quaternion<Real>::Quaternion(const Real* ptrq)
    : x_(ptrq[0]), y_(ptrq[1]), z_(ptrq[2]), w_(ptrq[3])
{
}

template <typename Real>
COMM_FUNC Quaternion<Real>::Quaternion(const Quaternion<Real>& quat)
    : x_(quat.x()), y_(quat.y()), z_(quat.z()), w_(quat.w())
{
}

template <typename Real>
COMM_FUNC Quaternion<Real>::Quaternion(const Vector<Real, 3>& euler_angle)
{
    setEulerAngle(euler_angle);
}

template <typename Real>
COMM_FUNC Quaternion<Real>& Quaternion<Real>::operator=(const Quaternion<Real>& quat)
{
    w_ = quat.w();
    x_ = quat.x();
    y_ = quat.y();
    z_ = quat.z();
    return *this;
}

template <typename Real>
COMM_FUNC Quaternion<Real>& Quaternion<Real>::operator+=(const Quaternion<Real>& quat)
{
    w_ += quat.w();
    x_ += quat.x();
    y_ += quat.y();
    z_ += quat.z();
    return *this;
}

template <typename Real>
COMM_FUNC Quaternion<Real>& Quaternion<Real>::operator-=(const Quaternion<Real>& quat)
{
    w_ -= quat.w();
    x_ -= quat.x();
    y_ -= quat.y();
    z_ -= quat.z();
    return *this;
}

template <typename Real>
COMM_FUNC Quaternion<Real> Quaternion<Real>::operator-(const Quaternion<Real>& quat) const
{
    return Quaternion(x_ - quat.x(), y_ - quat.y(), z_ - quat.z(), w_ - quat.w());
}

template <typename Real>
COMM_FUNC Quaternion<Real> Quaternion<Real>::operator-(void) const
{
    return Quaternion(-x_, -y_, -z_, -w_);
}

template <typename Real>
COMM_FUNC Quaternion<Real> Quaternion<Real>::operator+(const Quaternion<Real>& quat) const
{
    return Quaternion(x_ + quat.x(), y_ + quat.y(), z_ + quat.z(), w_ + quat.w());
}

template <typename Real>
COMM_FUNC Quaternion<Real> Quaternion<Real>::operator*(const Real& scale) const
{
    return Quaternion(x_ * scale, y_ * scale, z_ * scale, w_ * scale);
}

template <typename Real>
COMM_FUNC Quaternion<Real> Quaternion<Real>::operator*(const Quaternion<Real>& q) const
{
    Quaternion result;
    result.x_ = w_ * q.x_ + x_ * q.w_ + y_ * q.z_ - z_ * q.y_;
    result.y_ = w_ * q.y_ + y_ * q.w_ + z_ * q.x_ - x_ * q.z_;
    result.z_ = w_ * q.z_ + z_ * q.w_ + x_ * q.y_ - y_ * q.x_;
    result.w_ = w_ * q.w_ - x_ * q.x_ - y_ * q.y_ - z_ * q.z_;
    return result;
}
template <typename Real>
COMM_FUNC Quaternion<Real> Quaternion<Real>::multiply_q(const Quaternion<Real>& q)
{
    return Quaternion(w_ * q.x() + x_ * q.w() + y_ * q.z() - z_ * q.y(),
                      w_ * q.y() + y_ * q.w() + z_ * q.x() - x_ * q.z(),
                      w_ * q.z() + z_ * q.w() + x_ * q.y() - y_ * q.x(),
                      w_ * q.w() - x_ * q.x() - y_ * q.y() - z_ * q.z());
}

template <typename Real>
COMM_FUNC Quaternion<Real> Quaternion<Real>::operator/(const Real& scale) const
{
    return Quaternion(x_ / scale, y_ / scale, z_ / scale, w_ / scale);
}

template <typename Real>
COMM_FUNC bool Quaternion<Real>::operator==(const Quaternion<Real>& quat) const
{
    if (w_ == quat.w() && x_ == quat.x() && y_ == quat.y() && z_ == quat.z())
        return true;
    return false;
}

template <typename Real>
COMM_FUNC bool Quaternion<Real>::operator!=(const Quaternion<Real>& quat) const
{
    if (*this == quat)
        return false;
    return true;
}

template <typename Real>
COMM_FUNC Real& Quaternion<Real>::operator[](unsigned int idx)
{
    switch (idx)
    {
        case 0:
            return x_;
        case 1:
            return y_;
        case 2:
            return z_;
        case 3:
            return w_;
        default:
            return w_;
    }
}

template <typename Real>
COMM_FUNC const Real& Quaternion<Real>::operator[](unsigned int idx) const
{
    switch (idx)
    {
        case 0:
            return x_;
        case 1:
            return y_;
        case 2:
            return z_;
        case 3:
            return w_;
        default:
            return w_;
    }
}

template <typename Real>
COMM_FUNC Real Quaternion<Real>::norm()
{
    Real result = w_ * w_ + x_ * x_ + y_ * y_ + z_ * z_;
    result      = glm::sqrt(result);
    return result;
}

template <typename Real>
COMM_FUNC Quaternion<Real>& Quaternion<Real>::normalize()
{
    Real d = glm::sqrt(x_ * x_ + y_ * y_ + z_ * z_ + w_ * w_);
    if (d < 0.00001)
    {
        w_ = 1.0f;
        x_ = y_ = z_ = 0.0f;
        return *this;
    }
    d = 1.0 / d;
    x_ *= d;
    y_ *= d;
    z_ *= d;
    w_ *= d;
    return *this;
}

template <typename Real>
COMM_FUNC void Quaternion<Real>::setValue(const Vector<Real, 3>& vec3, Real scale)
{
    w_ = scale;
    x_ = vec3[0];
    y_ = vec3[1];
    z_ = vec3[2];
}

template <typename Real>
COMM_FUNC void Quaternion<Real>::setValue(Real scale, const Vector<Real, 3>& vec3)
{
    w_ = scale;
    x_ = vec3[0];
    y_ = vec3[1];
    z_ = vec3[2];
}

template <typename Real>
COMM_FUNC void Quaternion<Real>::set(const Vector<Real, 3>& euler_angle)
{
    Real cos_roll  = glm::cos(euler_angle[0] * Real(0.5));
    Real sin_roll  = glm::sin(euler_angle[0] * Real(0.5));
    Real cos_pitch = glm::cos(euler_angle[1] * Real(0.5));
    Real sin_pitch = glm::sin(euler_angle[1] * Real(0.5));
    Real cos_yaw   = glm::cos(euler_angle[2] * Real(0.5));
    Real sin_yaw   = glm::sin(euler_angle[2] * Real(0.5));

    w_ = cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw;
    x_ = cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw;
    y_ = cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw;
    z_ = sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw;
}

template <typename Real>
COMM_FUNC Vector<Real, 3> Quaternion<Real>::getEulerAngle() const
{
    Vector<Real, 3> euler_angle;
    euler_angle[0] = atan2(Real(2.0) * (w_ * z_ + x_ * y_), Real(1.0) - Real(2.0) * (z_ * z_ + x_ * x_));
    Real tmp       = (Real(2.0) * (w_ * x_ - y_ * z_));
    if (tmp > 1.0)
        tmp = 1.0;
    if (tmp < -1.0)
        tmp = -1.0;
    euler_angle[1] = glm::asin(tmp);
    euler_angle[2] = atan2(Real(2.0) * (w_ * y_ + z_ * x_), Real(1.0) - Real(2.0) * (x_ * x_ + y_ * y_));
    return euler_angle;
}

template <typename Real>
inline COMM_FUNC void Quaternion<Real>::setEulerAngle(const Vector<Real, 3>& euler_angle)
{
    Real cos_roll  = glm::cos(euler_angle[0] * Real(0.5));
    Real sin_roll  = glm::sin(euler_angle[0] * Real(0.5));
    Real cos_pitch = glm::cos(euler_angle[1] * Real(0.5));
    Real sin_pitch = glm::sin(euler_angle[1] * Real(0.5));
    Real cos_yaw   = glm::cos(euler_angle[2] * Real(0.5));
    Real sin_yaw   = glm::sin(euler_angle[2] * Real(0.5));

    x_ = cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw;
    y_ = cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw;
    z_ = sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw;
    w_ = cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw;
}

template <typename Real>
COMM_FUNC Real Quaternion<Real>::getAngle() const
{
    return glm::acos(w_) * (Real)(2);
}

template <typename Real>
COMM_FUNC Real Quaternion<Real>::getAngle(const Quaternion<Real>& quat) const
{
    return glm::acos(dot(quat)) * (Real)(2);
}

template <typename Real>
COMM_FUNC Real Quaternion<Real>::dot(const Quaternion<Real>& quat) const
{
    return w_ * quat.w() + x_ * quat.x() + y_ * quat.y() + z_ * quat.z();
}

template <typename Real>
COMM_FUNC Quaternion<Real> Quaternion<Real>::getConjugate() const
{
    return Quaternion<Real>(-x_, -y_, -z_, w_);
}

template <typename Real>
COMM_FUNC const Vector<Real, 3> Quaternion<Real>::rotate(const Vector<Real, 3> v) const
{
    const Real vx   = Real(2.0) * v[0];
    const Real vy   = Real(2.0) * v[1];
    const Real vz   = Real(2.0) * v[2];
    const Real w2   = w_ * w_ - ( Real )0.5;
    const Real dot2 = (x_ * vx + y_ * vy + z_ * vz);
    return Vector<Real, 3>(
        (vx * w2 + (y_ * vz - z_ * vy) * w_ + x_ * dot2),
        (vy * w2 + (z_ * vx - x_ * vz) * w_ + y_ * dot2),
        (vz * w2 + (x_ * vy - y_ * vx) * w_ + z_ * dot2));
}

template <typename Real>
COMM_FUNC void Quaternion<Real>::rotateVector(Vector<Real, 3>& v) const
{
    const Real vx   = Real(2.0) * v[0];
    const Real vy   = Real(2.0) * v[1];
    const Real vz   = Real(2.0) * v[2];
    const Real w2   = w_ * w_ - ( Real )0.5;
    const Real dot2 = (x_ * vx + y_ * vy + z_ * vz);
    v[0]            = vx * w2 + (y_ * vz - z_ * vy) * w_ + x_ * dot2;
    v[1]            = vy * w2 + (z_ * vx - x_ * vz) * w_ + y_ * dot2;
    v[2]            = vz * w2 + (x_ * vy - y_ * vx) * w_ + z_ * dot2;
}

template <typename Real>
COMM_FUNC SquareMatrix<Real, 3> Quaternion<Real>::get3x3Matrix() const
{
    Real x = x_, y = y_, z = z_, w = w_;
    Real x2 = x + x, y2 = y + y, z2 = z + z;
    Real xx = x2 * x, yy = y2 * y, zz = z2 * z;
    Real xy = x2 * y, xz = x2 * z, xw = x2 * w;
    Real yz = y2 * z, yw = y2 * w, zw = z2 * w;
    return SquareMatrix<Real, 3>(Real(1) - yy - zz, xy - zw, xz + yw, xy + zw, Real(1) - xx - zz, yz - xw, xz - yw, yz + xw, Real(1) - xx - yy);
}

template <typename Real>
COMM_FUNC SquareMatrix<Real, 4> Quaternion<Real>::get4x4Matrix() const
{
    Real x = x_, y = y_, z = z_, w = w_;
    Real x2 = x + x, y2 = y + y, z2 = z + z;
    Real xx = x2 * x, yy = y2 * y, zz = z2 * z;
    Real xy = x2 * y, xz = x2 * z, xw = x2 * w;
    Real yz = y2 * z, yw = y2 * w, zw = z2 * w;

    return SquareMatrix<Real, 4>(
        Real(1) - yy - zz, xy - zw, xz + yw, Real(0), xy + zw, Real(1) - xx - zz, yz - xw, Real(0), xz - yw, yz + xw, Real(1) - xx - yy, Real(0), Real(0), Real(0), Real(0), Real(1));
}

template <typename Real>
COMM_FUNC Quaternion<Real>::Quaternion(const SquareMatrix<Real, 3>& matrix)
{
    Real tr      = matrix(0, 0) + matrix(1, 1) + matrix(2, 2);
    int  i       = 3;
    Real maxdiag = tr;
    if (matrix(0, 0) > maxdiag)
    {
        i       = 0;
        maxdiag = matrix(0, 0);
    }
    if (matrix(1, 1) > maxdiag)
    {
        i       = 1;
        maxdiag = matrix(1, 1);
    }
    if (matrix(2, 2) > maxdiag)
    {
        i       = 2;
        maxdiag = matrix(2, 2);
    }
    if (i == 3)
    {
        Real s = glm::sqrt(tr + Real(1.0));
        w_     = s * Real(0.5);
        if (s != 0.0)
            s = Real(0.5) / s;
        x_ = s * (matrix(2, 1) - matrix(1, 2));
        y_ = s * (matrix(0, 2) - matrix(2, 0));
        z_ = s * (matrix(1, 0) - matrix(0, 1));
    }
    else
    {
        int j, k;
        int next[3] = { 1, 2, 0 };
        j           = next[i];
        k           = next[j];
        Real q[4];
        Real s = glm::sqrt(matrix(i, i) - matrix(j, j) - matrix(k, k) + Real(1.0));
        q[i]   = s * Real(0.5);
        s      = 0.25 / q[i];
        q[3]   = s * (matrix(k, j) - matrix(j, k));
        q[j]   = s * (matrix(i, j) + matrix(j, i));
        q[k]   = s * (matrix(i, k) + matrix(k, i));
        x_     = q[0];
        y_     = q[1];
        z_     = q[2];
        w_     = q[3];
    }
}

template <typename Real>
COMM_FUNC Quaternion<Real>::Quaternion(const SquareMatrix<Real, 4>& matrix)
{
    Real tr      = matrix(0, 0) + matrix(1, 1) + matrix(2, 2);
    int  i       = 3;
    Real maxdiag = tr;
    if (matrix(0, 0) > maxdiag)
    {
        i       = 0;
        maxdiag = matrix(0, 0);
    }
    if (matrix(1, 1) > maxdiag)
    {
        i       = 1;
        maxdiag = matrix(1, 1);
    }
    if (matrix(2, 2) > maxdiag)
    {
        i       = 2;
        maxdiag = matrix(2, 2);
    }
    if (i == 3)
    {
        Real s = glm::sqrt(tr + Real(1.0));
        w_     = s * Real(0.5);
        if (s != 0.0)
            s = Real(0.5) / s;
        x_ = s * (matrix(2, 1) - matrix(1, 2));
        y_ = s * (matrix(0, 2) - matrix(2, 0));
        z_ = s * (matrix(1, 0) - matrix(0, 1));
    }
    else
    {
        int j, k;
        int next[3] = { 1, 2, 0 };
        j           = next[i];
        k           = next[j];
        Real q[4];
        Real s = glm::sqrt(matrix(i, i) - matrix(j, j) - matrix(k, k) + Real(1.0));
        q[i]   = s * Real(0.5);
        s      = 0.25 / q[i];
        q[3]   = s * (matrix(k, j) - matrix(j, k));
        q[j]   = s * (matrix(i, j) + matrix(j, i));
        q[k]   = s * (matrix(i, k) + matrix(k, i));
        x_     = q[0];
        y_     = q[1];
        z_     = q[2];
        w_     = q[3];
    }
}

template <typename Real>
COMM_FUNC void Quaternion<Real>::getRotation(Real& rot, Vector<Real, 3>& axis) const
{
    rot = 2.0f * glm::acos(w_);
    if (rot == 0)
    {
        axis[0] = axis[1] = 0;
        axis[2]           = 1;
        return;
    }
    axis[0] = x_;
    axis[1] = y_;
    axis[2] = z_;
    axis.normalize();
}

template <typename Real>
inline COMM_FUNC void Quaternion<Real>::setRotation(const Vector<Real, 3>& unit_axis, const Real& radAng)
{
    Real a = radAng * ( Real )0.5;
    Real s = glm::sin(a) / unit_axis.norm();
    w_     = glm::cos(a);
    x_     = unit_axis[0] * s;
    y_     = unit_axis[1] * s;
    z_     = unit_axis[2] * s;
}

//explicit instantiation

}  // namespace PhysIKA
