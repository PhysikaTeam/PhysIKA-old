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
    w(1),
    x(0),
    y(0),
    z(0)
{

}

template <typename Scalar>
Quaternion<Scalar>::Quaternion(Scalar _x, Scalar _y, Scalar _z, Scalar _w):
    w(_w),
    x(_x),
    y(_y),
    z(_z)
{

}

template <typename Scalar>
Quaternion<Scalar>::Quaternion(const Vector3D<Scalar> & vec3, float _w):
    w(_w),
    x(vec3[0]),
    y(vec3[1]),
    z(vec3[2])
{
    
}
template <typename Scalar>
Quaternion<Scalar>::Quaternion(float _w, const Vector3D<Scalar> & vec3):
    w(_w),
    x(vec3[0]),
    y(vec3[1]),
    z(vec3[2])
{

}

template <typename Scalar>
Quaternion<Scalar>::Quaternion(const Scalar *ptrq):
    w(ptrq[4]),
    x(ptrq[0]),
    y(ptrq[1]),
    z(ptrq[2])
{

}

template <typename Scalar>
Quaternion<Scalar>::Quaternion(const Quaternion<Scalar> & quat):
    w(quat.w),
    x(quat.x),
    y(quat.y),
    z(quat.z)
{

}

template <typename Scalar>
Quaternion<Scalar> & Quaternion<Scalar>::operator = (const Quaternion<Scalar> &quat)
{
    w = quat.w;
    x = quat.x;
    y = quat.y;
    z = quat.z;
    return *this;
}

template <typename Scalar>
Quaternion<Scalar> & Quaternion<Scalar>::operator += (const Quaternion<Scalar> &quat)
{
    w += quat.w;
    x += quat.x;
    y += quat.y;
    z += quat.z;
    return *this;
}

template <typename Scalar>
Quaternion<Scalar> & Quaternion<Scalar>::operator -= (const Quaternion<Scalar> &quat)
{
    w -= quat.w;
    x -= quat.x;
    y -= quat.y;
    z -= quat.z;
    return *this;
}

template <typename Scalar>
Quaternion<Scalar>  Quaternion<Scalar>::operator - (const Quaternion<Scalar> &quat)
{
    return Quaternion(x-quat.x,y-quat.y,z-quat.z,w-quat.w);
}

template <typename Scalar>
Quaternion<Scalar>  Quaternion<Scalar>::operator - (void)
{
    return Quaternion(-x,-y,-z,-w);
}

template <typename Scalar>
Quaternion<Scalar>  Quaternion<Scalar>::operator + (const Quaternion<Scalar> &quat)
{
    return Quaternion(x+quat.x,y+quat.y,z+quat.z,w+quat.w);
}

template <typename Scalar>
Quaternion<Scalar>  Quaternion<Scalar>::operator * (Scalar scale)
{
    return Quaternion(x*scale,y*scale,z*scale,w*scale);
}

template <typename Scalar>
Quaternion<Scalar>  Quaternion<Scalar>::operator / (Scalar scale)
{
    assert(scale != 0);
    return Quaternion(x/scale,y/scale,z/scale,w/scale);
}

template <typename Scalar>
bool  Quaternion<Scalar>::operator == (const Quaternion<Scalar> &quat)
{
    if(w == quat.w && x == quat.x && y == quat.y && z == quat.z)
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
        return x;
    case 1:
        return y;
    case 2:
        return z;
    case 3:
        return w;
    default:
        return w;
    }
}

template <typename Scalar>
const Scalar&  Quaternion<Scalar>::operator[] (int idx) const
{
    assert(idx >= 0 && idx <= 3);
    switch(idx){
    case 0:
        return x;
    case 1:
        return y;
    case 2:
        return z;
    case 3:
        return w;
    default:
        return w;
    }
}

template <typename Scalar>
Scalar Quaternion<Scalar>::norm()
{
    Scalar result = w*w + x*x + y*y + z*z;
    result = sqrt(result);
    return result;
}

template <typename Scalar>
Quaternion<Scalar>& Quaternion<Scalar>::normalize()
{
    Scalar norm = this->norm();
    if(norm)
    {
        w/=norm;
        x/=norm;
        y/=norm;
        z/=norm;
    }
    return *this;
}


template <typename Scalar>
void Quaternion<Scalar>::set(const Vector3D<Scalar>& vec3, Scalar scale)
{
     w = scale;
    x = vec3[0];
    y = vec3[1];
    z = vec3[2];
}

template <typename Scalar>
void Quaternion<Scalar>::set(Scalar scale, const Vector3D<Scalar>& vec3)
{
    w = scale;
    x = vec3[0];
    y = vec3[1];
    z = vec3[2];
}





template class Quaternion<float>;
template class Quaternion<double>;


}