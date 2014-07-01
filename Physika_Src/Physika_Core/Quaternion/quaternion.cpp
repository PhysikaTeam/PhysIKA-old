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
Quaternion<Scalar>  Quaternion<Scalar>::operator - (const Quaternion<Scalar> &quat)
{
    return Quaternion(x_ - quat.x(), y_ - quat.y(), z_ - quat.z(), w_ - quat.w());
}

template <typename Scalar>
Quaternion<Scalar>  Quaternion<Scalar>::operator - (void)
{
    return Quaternion(-x_, -y_, -z_, -w_);
}

template <typename Scalar>
Quaternion<Scalar>  Quaternion<Scalar>::operator + (const Quaternion<Scalar> &quat)
{
    return Quaternion(x_ + quat.x(), y_ + quat.y(), z_ + quat.z(), w_ + quat.w());
}

template <typename Scalar>
Quaternion<Scalar>  Quaternion<Scalar>::operator * (Scalar scale)
{
    return Quaternion(x_ * scale, y_ * scale, z_ * scale, w_ * scale);
}

template <typename Scalar>
Quaternion<Scalar>  Quaternion<Scalar>::operator / (Scalar scale)
{
    if(abs(scale)<std::numeric_limits<Scalar>::epsilon())
    {
	std::cerr<<"Quaternion Divide by zero error!\n";
	std::exit(EXIT_FAILURE);
    }
    return Quaternion(x_ / scale, y_ / scale, z_ / scale, w_ / scale);
}

template <typename Scalar>
bool  Quaternion<Scalar>::operator == (const Quaternion<Scalar> &quat)
{
    if (w_ == quat.w() && x_ == quat.x() && y_ == quat.y() && z_ == quat.z())
        return true;
    return false;
}

template <typename Scalar>
bool  Quaternion<Scalar>::operator != (const Quaternion<Scalar> &quat)
{
    if (*this == quat)
        return false;
    return true;
}

template <typename Scalar>
Scalar&  Quaternion<Scalar>::operator[] (int idx)
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
const Scalar&  Quaternion<Scalar>::operator[] (int idx) const
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
        Scalar s = sqrt(tr + 1.0);
        w_ = s * 0.5;
        if(s != 0.0)
            s = 0.5 / s;
        x_ = s * (matrix(1,2) - matrix(2,1));
        y_ = s * (matrix(2,0) - matrix(0,2));
        z_ = s * (matrix(0,1) - matrix(1,0));
    }
    else
    {
        int i = 0, j, k;
        int next[3] = { 1, 2, 0 }; 
        int q[4];
        if(matrix(1,1) > matrix(0,0)) i = 1;
        if(matrix(2,2) > matrix(i,i)) i = 2;
        j = next[i];
        k = next[j];
        Scalar s = sqrt(matrix(i,i) - matrix(j,j) - matrix(k,k) + 1.0);
        q[i] = s * 0.5;
        if(s != 0.0) 
            s = 0.5/s;
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
        Scalar s = sqrt(tr + 1.0);
        w_ = s * 0.5;
        if(s != 0.0)
            s = 0.5 / s;
        x_ = s * (matrix(1,2) - matrix(2,1));
        y_ = s * (matrix(2,0) - matrix(0,2));
        z_ = s * (matrix(0,1) - matrix(1,0));
    }
    else
    {
        int i = 0, j, k;
        int next[3] = { 1, 2, 0 }; 
        int q[4];
        if(matrix(1,1) > matrix(0,0)) i = 1;
        if(matrix(2,2) > matrix(i,i)) i = 2;
        j = next[i];
        k = next[j];
        Scalar s = sqrt(matrix(i,i) - matrix(j,j) - matrix(k,k) + 1.0);
        q[i] = s * 0.5;
        if(s != 0.0) 
            s = 0.5/s;
        q[3] = s * (matrix(j,k) - matrix(k,j));
        q[j] = s * (matrix(i,j) - matrix(j,i));
        q[k] = s * (matrix(i,k) - matrix(k,i));
        x_ = q[0];
        y_ = q[1];
        z_ = q[2];
        w_ = q[3];
    }
}
//explicit instantiation
template class Quaternion<float>;
template class Quaternion<double>;

}
