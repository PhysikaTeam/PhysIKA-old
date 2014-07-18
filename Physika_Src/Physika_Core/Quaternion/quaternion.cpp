/*
 * @file quaternion.cpp 
 * @brief quaternion.
 * @author Sheng Yang, Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <cmath>
#include <cstdlib>
#include <iostream>
#include "Physika_Core/Quaternion/quaternion.h"
#include "Physika_Core/Utilities/math_utilities.h"

namespace Physika{

template <typename Scalar>
Quaternion<Scalar>::Quaternion():
        w_(1),
        x_(0),
        y_(0),
        z_(0)
{

}

template <typename Scalar>
Quaternion<Scalar>::Quaternion(Scalar x, Scalar y, Scalar z, Scalar w):
        w_(w),
        x_(x),
        y_(y),
        z_(z)
{

}

template <typename Scalar>
Quaternion<Scalar>::Quaternion(const Vector<Scalar,3> & unit_axis, Scalar angle_rad)
{
    const Scalar a = angle_rad * (Scalar)0.5;
    const Scalar s = sin(a);
    w_ = cos(a);
    x_ = unit_axis[0] * s;
    y_ = unit_axis[1] * s;
    z_ = unit_axis[2] * s;
}

template <typename Scalar>
Quaternion<Scalar>::Quaternion(Scalar angle_rad, const Vector<Scalar,3> & unit_axis)
{
    const Scalar a = angle_rad * (Scalar)0.5;
    const Scalar s = sin(a);
    w_ = cos(a);
    x_ = unit_axis[0] * s;
    y_ = unit_axis[1] * s;
    z_ = unit_axis[2] * s;
}

template <typename Scalar>
Quaternion<Scalar>::Quaternion(const Scalar *ptrq):
    w_(ptrq[4]),
    x_(ptrq[0]),
    y_(ptrq[1]),
    z_(ptrq[2])
{

}

template <typename Scalar>
Quaternion<Scalar>::Quaternion(const Quaternion<Scalar> & quat):
    w_(quat.w()),
    x_(quat.x()),
    y_(quat.y()),
    z_(quat.z())
{

}

template <typename Scalar>
Quaternion<Scalar>:: Quaternion(const Vector<Scalar, 3>& euler_angle)
{
    Scalar cos_roll = cos(euler_angle[0] * Scalar(0.5));
    Scalar sin_roll = sin(euler_angle[0] * Scalar(0.5));
    Scalar cos_pitch = cos(euler_angle[1] * Scalar(0.5));
    Scalar sin_pitch = sin(euler_angle[1] * Scalar(0.5));
    Scalar cos_yaw = cos(euler_angle[2] * Scalar(0.5));
    Scalar sin_yaw = sin(euler_angle[2] * Scalar(0.5));

    w_ = cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw;
    x_ = cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw;
    y_ = cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw;
    z_ = sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw;
}


template <typename Scalar>
Quaternion<Scalar> & Quaternion<Scalar>::operator = (const Quaternion<Scalar> &quat)
{
    w_ = quat.w();
    x_ = quat.x();
    y_ = quat.y();
    z_ = quat.z();
    return *this;
}

template <typename Scalar>
Quaternion<Scalar> & Quaternion<Scalar>::operator += (const Quaternion<Scalar> &quat)
{
    w_ += quat.w();
    x_ += quat.x();
    y_ += quat.y();
    z_ += quat.z();
    return *this;
}

template <typename Scalar>
Quaternion<Scalar> & Quaternion<Scalar>::operator -= (const Quaternion<Scalar> &quat)
{
    w_ -= quat.w();
    x_ -= quat.x();
    y_ -= quat.y();
    z_ -= quat.z();
    return *this;
}

template <typename Scalar>
Quaternion<Scalar>  Quaternion<Scalar>::operator - (const Quaternion<Scalar> &quat) const
{
    return Quaternion(x_ - quat.x(), y_ - quat.y(), z_ - quat.z(), w_ - quat.w());
}

template <typename Scalar>
Quaternion<Scalar>  Quaternion<Scalar>::operator - (void) const
{
    return Quaternion(-x_, -y_, -z_, -w_);
}

template <typename Scalar>
Quaternion<Scalar>  Quaternion<Scalar>::operator + (const Quaternion<Scalar> &quat) const
{
    return Quaternion(x_ + quat.x(), y_ + quat.y(), z_ + quat.z(), w_ + quat.w());
}

template <typename Scalar>
Quaternion<Scalar>  Quaternion<Scalar>::operator * (const Scalar& scale) const
{
    return Quaternion(x_ * scale, y_ * scale, z_ * scale, w_ * scale);
}

template <typename Scalar>
Quaternion<Scalar> Quaternion<Scalar>::operator * (const Quaternion<Scalar>& q) const
{
    return Quaternion(  w_ * q.x() + x_ * q.w() + y_ * q.z() - z_ * q.y(),
                        w_ * q.y() + y_ * q.w() + z_ * q.x() - x_ * q.z(),
                        w_ * q.z() + z_ * q.w() + x_ * q.y() - y_ * q.x(),
                        w_ * q.w() - x_ * q.x() - y_ * q.y() - z_ * q.z());
}


template <typename Scalar>
Quaternion<Scalar>  Quaternion<Scalar>::operator / (const Scalar& scale) const
{
    if(abs(scale)<std::numeric_limits<Scalar>::epsilon())
    {
	std::cerr<<"Quaternion Divide by zero error!\n";
	std::exit(EXIT_FAILURE);
    }
    return Quaternion(x_ / scale, y_ / scale, z_ / scale, w_ / scale);
}

template <typename Scalar>
bool  Quaternion<Scalar>::operator == (const Quaternion<Scalar> &quat) const
{
    if (w_ == quat.w() && x_ == quat.x() && y_ == quat.y() && z_ == quat.z())
        return true;
    return false;
}

template <typename Scalar>
bool  Quaternion<Scalar>::operator != (const Quaternion<Scalar> &quat) const
{
    if (*this == quat)
        return false;
    return true;
}

template <typename Scalar>
Scalar&  Quaternion<Scalar>::operator[] (unsigned int idx)
{
    if(idx < 0 || idx > 3)
    {
        std::cerr<<"Quaternion index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    switch(idx)
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

template <typename Scalar>
const Scalar&  Quaternion<Scalar>::operator[] (unsigned int idx) const
{
    if(idx < 0 || idx > 3)
    {
        std::cerr<<"Quaternion index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    switch(idx)
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

template <typename Scalar>
Scalar Quaternion<Scalar>::norm()
{
    Scalar result = w_ * w_ + x_ * x_ + y_ * y_ + z_ * z_;
    result = sqrt(result);
    return result;
}

template <typename Scalar>
Quaternion<Scalar>& Quaternion<Scalar>::normalize()
{
    Scalar norm = this->norm();
    if (norm)
    {
        w_/=norm;
        x_/=norm;
        y_/=norm;
        z_/=norm;
    }
    return *this;
}

template <typename Scalar>
void Quaternion<Scalar>::set(const Vector<Scalar,3>& vec3, Scalar scale)
{
    w_ = scale;
    x_ = vec3[0];
    y_ = vec3[1];
    z_ = vec3[2];
}

template <typename Scalar>
void Quaternion<Scalar>::set(Scalar scale, const Vector<Scalar,3>& vec3)
{
    w_ = scale;
    x_ = vec3[0];
    y_ = vec3[1];
    z_ = vec3[2];
}

template <typename Scalar>
void Quaternion<Scalar>::set(const Vector<Scalar,3>& euler_angle)
{
    Scalar cos_roll = cos(euler_angle[0] * Scalar(0.5));
    Scalar sin_roll = sin(euler_angle[0] * Scalar(0.5));
    Scalar cos_pitch = cos(euler_angle[1] * Scalar(0.5));
    Scalar sin_pitch = sin(euler_angle[1] * Scalar(0.5));
    Scalar cos_yaw = cos(euler_angle[2] * Scalar(0.5));
    Scalar sin_yaw = sin(euler_angle[2] * Scalar(0.5));

    w_ = cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw;
    x_ = cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw;
    y_ = cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw;
    z_ = sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw;
}

template <typename Scalar>
Vector<Scalar, 3> Quaternion<Scalar>::getEulerAngle() const
{
    Vector<Scalar, 3> euler_angle;
    euler_angle[0] = atan2(Scalar(2.0) * (w_ * z_ + x_ * y_), Scalar(1.0) - Scalar(2.0) * (z_ * z_ + x_ * x_));
    Scalar tmp = (Scalar(2.0) * (w_ * x_ - y_ * z_));
    if(tmp > 1.0)
        tmp = 1.0;
    if(tmp < -1.0)
        tmp = -1.0;
    euler_angle[1] = asin(tmp);
    euler_angle[2] = atan2(Scalar(2.0) * (w_ * y_ + z_ * x_), Scalar(1.0) - Scalar(2.0) * (x_ * x_ + y_ * y_));
    return euler_angle;
}

template <typename Scalar>
Scalar Quaternion<Scalar>::getAngle() const
{
    return acos(w_) * (Scalar)(2);
}

template <typename Scalar>
Scalar Quaternion<Scalar>::getAngle(const Quaternion<Scalar>& quat) const
{
    return acos(dot(quat)) * (Scalar)(2);
}

template <typename Scalar>
Scalar Quaternion<Scalar>::dot(const Quaternion<Scalar> & quat) const
{
    return w_ * quat.w() + x_ * quat.x() + y_ * quat.y() + z_ * quat.z ();
}

template <typename Scalar>
Quaternion<Scalar> Quaternion<Scalar>::getConjugate() const
{
    return Quaternion<Scalar>(-x_, -y_, -z_, w_);
}

template <typename Scalar>
const Vector<Scalar,3> Quaternion<Scalar>::rotate(const Vector<Scalar,3> v) const 
{
    const Scalar vx = Scalar(2.0) * v[0];
    const Scalar vy = Scalar(2.0) * v[1];
    const Scalar vz = Scalar(2.0) * v[2];
    const Scalar w2 = w_ * w_ - (Scalar)0.5;
    const Scalar dot2 = (x_ * vx + y_ * vy + z_ * vz);
    return Vector<Scalar,3>
        (
        (vx * w2 + (y_ * vz - z_ * vy) * w_ + x_ * dot2), 
        (vy * w2 + (z_ * vx - x_ * vz) * w_ + y_ * dot2), 
        (vz * w2 + (x_ * vy - y_ * vx) * w_ + z_ * dot2)
        );
}


template <typename Scalar>
SquareMatrix<Scalar, 3> Quaternion<Scalar>::get3x3Matrix() const
{
    Scalar x = x_, y = y_, z = z_, w = w_;
    Scalar x2 = x + x, y2 = y + y, z2 = z + z;
    Scalar xx = x2 * x, yy = y2 * y, zz = z2 * z;
    Scalar xy = x2 * y, xz = x2 * z, xw = x2 * w;
    Scalar yz = y2 * z, yw = y2 * w, zw = z2 * w;
    return SquareMatrix<Scalar, 3>(Scalar(1) - yy - zz, xy - zw, xz + yw,
                                    xy + zw, Scalar(1) - xx - zz, yz - xw,
                                   xz - yw, yz + xw, Scalar(1) - xx - yy);
}

template <typename Scalar>
SquareMatrix<Scalar,4> Quaternion<Scalar>::get4x4Matrix() const
{
    Scalar x = x_, y = y_, z = z_, w = w_;
    Scalar x2 = x + x, y2 = y + y, z2 = z + z;
    Scalar xx = x2 * x, yy = y2 * y, zz = z2 * z;
    Scalar xy = x2 * y, xz = x2 * z, xw = x2 * w;
    Scalar yz = y2 * z, yw = y2 * w, zw = z2 * w;
    Scalar entries[16];
    entries[0] = Scalar(1) - yy - zz;
    entries[1] = xy - zw;
    entries[2] = xz + yw,
    entries[3] = 0;
    entries[4] = xy + zw;
    entries[5] = Scalar(1) - xx - zz;
    entries[6] = yz - xw;
    entries[7] = 0;
    entries[8] = xz - yw;
    entries[9] = yz + xw;
    entries[10] = Scalar(1) - xx - yy;
    entries[11] = 0;
    entries[12] = 0;
    entries[13] = 0;
    entries[14] = 0;
    entries[15] = 1;
    return SquareMatrix<Scalar,4>(entries[0],entries[1],entries[2],entries[3],
                                  entries[4],entries[5],entries[6],entries[7],
                                  entries[8],entries[9],entries[10],entries[11],
                                  entries[12],entries[13],entries[14],entries[15]);

}

template <typename Scalar>
Quaternion<Scalar>::Quaternion(const SquareMatrix<Scalar, 3>& matrix )
{
    Scalar tr = matrix(0,0) + matrix(1,1) + matrix(2,2);
    if(tr > 0.0)
    {
        Scalar s = sqrt(tr + Scalar(1.0));
        w_ = s * Scalar(0.5);
        if(s != 0.0)
            s = Scalar(0.5) / s;
        x_ = s * (matrix(1,2) - matrix(2,1));
        y_ = s * (matrix(2,0) - matrix(0,2));
        z_ = s * (matrix(0,1) - matrix(1,0));
    }
    else
    {
        int i = 0, j, k;
        int next[3] = { 1, 2, 0 }; 
        Scalar q[4];
        if(matrix(1,1) > matrix(0,0)) i = 1;
        if(matrix(2,2) > matrix(i,i)) i = 2;
        j = next[i];
        k = next[j];
        Scalar s = sqrt(matrix(i,i) - matrix(j,j) - matrix(k,k) + Scalar(1.0));
        q[i] = s * Scalar(0.5);
        if(s != 0.0) 
            s = Scalar(0.5)/s;
        q[3] = s * (matrix(j,k) - matrix(k,j));
        q[j] = s * (matrix(i,j) - matrix(j,i));
        q[k] = s * (matrix(i,k) - matrix(k,i));
        x_ = q[0];
        y_ = q[1];
        z_ = q[2];
        w_ = q[3];
    }
}

template <typename Scalar>
Quaternion<Scalar>::Quaternion(const SquareMatrix<Scalar,4>& matrix)
{
    Scalar tr = matrix(0,0) + matrix(1,1) + matrix(2,2);
    if(tr > 0.0)
    {
        Scalar s = sqrt(tr + Scalar(1.0));
        w_ = s * Scalar(0.5);
        if(s != 0.0)
            s = Scalar(0.5) / s;
        x_ = s * (matrix(1,2) - matrix(2,1));
        y_ = s * (matrix(2,0) - matrix(0,2));
        z_ = s * (matrix(0,1) - matrix(1,0));
    }
    else
    {
        int i = 0, j, k;
        int next[3] = { 1, 2, 0 }; 
        Scalar q[4];
        if(matrix(1,1) > matrix(0,0)) i = 1;
        if(matrix(2,2) > matrix(i,i)) i = 2;
        j = next[i];
        k = next[j];
        Scalar s = sqrt(matrix(i,i) - matrix(j,j) - matrix(k,k) + Scalar(1.0));
        q[i] = s * Scalar(0.5);
        if(s != 0.0) 
            s = Scalar(0.5)/s;
        q[3] = s * (matrix(j,k) - matrix(k,j));
        q[j] = s * (matrix(i,j) - matrix(j,i));
        q[k] = s * (matrix(i,k) - matrix(k,i));
        x_ = q[0];
        y_ = q[1];
        z_ = q[2];
        w_ = q[3];
    }
}

template <typename Scalar>
void Quaternion<Scalar>::toRadiansAndUnitAxis(Scalar& angle, Vector<Scalar, 3>& axis) const
{
    const Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
    const Scalar s2 = x_*x_ + y_*y_ + z_*z_;
    if(s2 < epsilon*epsilon)
    {
        angle = 0;
        axis = Vector<Scalar, 3>(0,0,0);
    }
    else
    {
        const Scalar s =  1/(sqrt(s2));
        axis = Vector<Scalar,3>(x_, y_, z_) * s;
        angle = w_ < epsilon ? Scalar(PI):atan2(s2*s, w_) * 2;
    }
}

//explicit instantiation
template class Quaternion<float>;
template class Quaternion<double>;

}
