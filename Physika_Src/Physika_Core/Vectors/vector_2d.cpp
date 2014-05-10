/*
 * @file vector_2d.cpp 
 * @brief 2d vector.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <limits>
#include "Physika_Core/Utilities/Math_Utilities/math_utilities.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Vectors/vector_2d.h"

namespace Physika{

template <typename Scalar>
Vector<Scalar,2>::Vector()
{
}

template <typename Scalar>
Vector<Scalar,2>::Vector(Scalar x, Scalar y)
{
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    eigen_vector_2x_(0)=x;
    eigen_vector_2x_(1)=y;
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    data_[0]=x;
    data_[1]=y;
#endif
}

template <typename Scalar>
Vector<Scalar,2>::Vector(Scalar x)
{
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    eigen_vector_2x_(0)=x;
    eigen_vector_2x_(1)=x;
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    data_[0]=data_[1]=x;
#endif
}

template <typename Scalar>
Vector<Scalar,2>::Vector(const Vector<Scalar,2> &vec2)
{
    *this = vec2;
}

template <typename Scalar>
Vector<Scalar,2>::~Vector()
{
}

template <typename Scalar>
Scalar& Vector<Scalar,2>::operator[] (int idx)
{
    PHYSIKA_ASSERT(idx>=0&&idx<(*this).dims());
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    return eigen_vector_2x_(idx);
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    return data_[idx];
#endif
}

template <typename Scalar>
const Scalar& Vector<Scalar,2>::operator[] (int idx) const
{
    PHYSIKA_ASSERT(idx>=0&&idx<(*this).dims());
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    return eigen_vector_2x_(idx);
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    return data_[idx];
#endif
}

template <typename Scalar>
Vector<Scalar,2> Vector<Scalar,2>::operator+ (const Vector<Scalar,2> &vec2) const
{
    Scalar result[2];
    for(int i = 0; i < 2; ++i)
        result[i] = (*this)[i] + vec2[i];
    return Vector<Scalar,2>(result[0],result[1]);
}

template <typename Scalar>
Vector<Scalar,2>& Vector<Scalar,2>::operator+= (const Vector<Scalar,2> &vec2)
{
    for(int i = 0; i < 2; ++i)
        (*this)[i] = (*this)[i] + vec2[i];
    return *this;
}

template <typename Scalar>
Vector<Scalar,2> Vector<Scalar,2>::operator- (const Vector<Scalar,2> &vec2) const
{
    Scalar result[2];
    for(int i = 0; i < 2; ++i)
        result[i] = (*this)[i] - vec2[i];
    return Vector<Scalar,2>(result[0],result[1]);
}

template <typename Scalar>
Vector<Scalar,2>& Vector<Scalar,2>::operator-= (const Vector<Scalar,2> &vec2)
{
    for(int i = 0; i < 2; ++i)
        (*this)[i] = (*this)[i] - vec2[i];
    return *this;
}

template <typename Scalar>
Vector<Scalar,2>& Vector<Scalar,2>::operator= (const Vector<Scalar,2> &vec2)
{
    for(int i = 0; i < 2; ++i)
        (*this)[i] = vec2[i];
    return *this;
}

template <typename Scalar>
bool Vector<Scalar,2>::operator== (const Vector<Scalar,2> &vec2) const
{
    for(int i = 0; i < 2; ++i)
        if((*this)[i] != vec2[i])
            return false;
    return true;
}

template <typename Scalar>
Vector<Scalar,2> Vector<Scalar,2>::operator* (Scalar scale) const
{
    Scalar result[2];
    for(int i = 0; i < 2; ++i)
        result[i] = (*this)[i] * scale;
    return Vector<Scalar,2>(result[0],result[1]);
}

template <typename Scalar>
Vector<Scalar,2>& Vector<Scalar,2>::operator*= (Scalar scale)
{
    for(int i = 0; i < 2; ++i)
        (*this)[i] = (*this)[i] * scale;
    return *this;
}

template <typename Scalar>
Vector<Scalar,2> Vector<Scalar,2>::operator/ (Scalar scale) const
{
    PHYSIKA_ASSERT(abs(scale)>std::numeric_limits<Scalar>::epsilon());
    Scalar result[2];
    for(int i = 0; i < 2; ++i)
        result[i] = (*this)[i] / scale;
    return Vector<Scalar,2>(result[0],result[1]);
}

template <typename Scalar>
Vector<Scalar,2>& Vector<Scalar,2>::operator/= (Scalar scale)
{
    PHYSIKA_ASSERT(abs(scale)>std::numeric_limits<Scalar>::epsilon());
    for(int i = 0; i < 2; ++i)
        (*this)[i] = (*this)[i] / scale;
    return *this;
}

template <typename Scalar>
Scalar Vector<Scalar,2>::norm() const
{
    Scalar result = (*this)[0]*(*this)[0] + (*this)[1]*(*this)[1];
    result = sqrt(result);
    return result;
}

template <typename Scalar>
Vector<Scalar,2>& Vector<Scalar,2>::normalize()
{
    Scalar norm = (*this).norm();
    bool nonzero_norm = norm > std::numeric_limits<Scalar>::epsilon();
    if(nonzero_norm)
    {
        for(int i = 0; i < 2; ++i)
        (*this)[i] = (*this)[i] / norm;
    }
    return *this;
}

template <typename Scalar>
Scalar Vector<Scalar,2>::cross(const Vector<Scalar,2>& vec2) const
{
  return (*this)[0]*vec2[1] - (*this)[1]*vec2[0];
}

template <typename Scalar>
Vector<Scalar,2> Vector<Scalar,2>::operator-(void) const
{
    return Vector<Scalar,2>(-(*this)[0],-(*this)[1]);
}

template <typename Scalar>
Scalar Vector<Scalar,2>::dot(const Vector<Scalar,2>& vec2) const
{
    return (*this)[0]*vec2[0] + (*this)[1]*vec2[1];
}

//explicit instantiation of template so that it could be compiled into a lib
template class Vector<float,2>;
template class Vector<double,2>;
template class Vector<int,2>;

} //end of namespace Physika
