/*
 * @file quaternion.cpp 
 * @brief quaternion.
 * @author Sheng Yang
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
#include "Physika_Core/Quaternion/quaternion.h"
#include "Physika_Core/Utilities/math_constants.h"

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
Quaternion<Scalar>::Quaternion(const Vector<Scalar,3> & unitAxis, Scalar angleRadians)
{
    const Scalar a = angleRadians * (Scalar)0.5;
    const Scalar s = sin(a);
    w_ = cos(a);
    x_ = unitAxis[0] * s;
    y_ = unitAxis[1] * s;
    z_ = unitAxis[2] * s;
}

template <typename Scalar>
Quaternion<Scalar>::Quaternion(Scalar angleRadians, const Vector<Scalar,3> & unitAxis)
{
    const Scalar a = angleRadians * (Scalar)0.5;
    const Scalar s = sin(a);
    w_ = cos(a);
    x_ = unitAxis[0] * s;
    y_ = unitAxis[1] * s;
    z_ = unitAxis[2] * s;
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
    return Quaternion(x_-quat.x(),y_-quat.y(),z_-quat.z(),w_-quat.w());
}

template <typename Scalar>
Quaternion<Scalar>  Quaternion<Scalar>::operator - (void)
{
    return Quaternion(-x_,-y_,-z_,-w_);
}

template <typename Scalar>
Quaternion<Scalar>  Quaternion<Scalar>::operator + (const Quaternion<Scalar> &quat)
{
    return Quaternion(x_+quat.x(),y_+quat.y(),z_+quat.z(),w_+quat.w());
}

template <typename Scalar>
Quaternion<Scalar>  Quaternion<Scalar>::operator * (Scalar scale)
{
    return Quaternion(x_*scale,y_*scale,z_*scale,w_*scale);
}

template <typename Scalar>
Quaternion<Scalar>  Quaternion<Scalar>::operator / (Scalar scale)
{
    assert(scale != 0);
    return Quaternion(x_/scale,y_/scale,z_/scale,w_/scale);
}

template <typename Scalar>
bool  Quaternion<Scalar>::operator == (const Quaternion<Scalar> &quat)
{
    if(w_ == quat.w() && x_ == quat.x() && y_ == quat.y() && z_ == quat.z())
        return true;
    return false;
}

template <typename Scalar>
bool  Quaternion<Scalar>::operator != (const Quaternion<Scalar> &quat)
{
    if(*this == quat)
        return false;
    return true;
}

template <typename Scalar>
Scalar&  Quaternion<Scalar>::operator[] (int idx)
{
    assert(idx >= 0 && idx <= 3);
    switch(idx){
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
    assert(idx >= 0 && idx <= 3);
    switch(idx){
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
    Scalar result = w_*w_ + x_*x_ + y_*y_ + z_*z_;
    result = sqrt(result);
    return result;
}

template <typename Scalar>
Quaternion<Scalar>& Quaternion<Scalar>::normalize()
{
    Scalar norm = this->norm();
    if(norm)
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
    return w_*quat.w() + x_*quat.x() + y_*quat.y() + z_*quat.z ();
}

template <typename Scalar>
Quaternion<Scalar> Quaternion<Scalar>::getConjugate() const
{
    return Quaternion<Scalar>(-x_,-y_,-z_,w_);
}

template <typename Scalar>
const Vector<Scalar,3> Quaternion<Scalar>::rotate(const Vector<Scalar,3> v) const 
{
    const Scalar vx = Scalar(2.0) * v[0];
    const Scalar vy = Scalar(2.0) * v[1];
    const Scalar vz = Scalar(2.0) * v[2];
    const Scalar w2 = w_*w_ - (Scalar)0.5;
    const Scalar dot2 = (x_ * vx + y_ * vy + z_ * vz);
    return Vector<Scalar,3>
    (
        (vx*w2 + (y_ * vz - z_ * vy)*w_ + x_*dot2), 
        (vy*w2 + (z_ * vx - x_ * vz)*w_ + y_*dot2), 
        (vz*w2 + (x_ * vy - y_ * vx)*w_ + z_*dot2)
    );
}

//explicit instantiation
template class Quaternion<float>;
template class Quaternion<double>;

}
