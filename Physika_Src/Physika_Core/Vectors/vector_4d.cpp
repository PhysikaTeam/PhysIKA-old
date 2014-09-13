/*
 * @file vector_4d.cpp 
 * @brief 4d vector.
 * @author Liyou Xu, Fei Zhu
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
#include <cstdlib>
#include <iostream>
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Matrices/matrix_4x4.h"
#include "Physika_Core/Vectors/vector_4d.h"

namespace Physika{

template <typename Scalar>
Vector<Scalar,4>::Vector()
{
}

template <typename Scalar>
Vector<Scalar,4>::Vector(Scalar x, Scalar y, Scalar z, Scalar w)
{
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    eigen_vector_4x_(0)=x;
    eigen_vector_4x_(1)=y;
    eigen_vector_4x_(2)=z;
    eigen_vector_4x_(3)=w;
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    data_[0]=x;
    data_[1]=y;
    data_[2]=z;
    data_[3]=w;
#endif
}

template <typename Scalar>
Vector<Scalar,4>::Vector(Scalar x)
{
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    eigen_vector_4x_(0)=x;
    eigen_vector_4x_(1)=x;
    eigen_vector_4x_(2)=x;
    eigen_vector_4x_(3)=x;
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    data_[0]=data_[1]=data_[2]=data_[3]=x;
#endif
}

template <typename Scalar>
Vector<Scalar,4>::Vector(const Vector<Scalar,4> &vec4)
{
    *this = vec4;
}

template <typename Scalar>
Vector<Scalar,4>::~Vector()
{
}

template <typename Scalar>
Scalar& Vector<Scalar,4>::operator[] (unsigned int idx)
{
    if(idx>=4)
    {
        std::cout<<"Vector index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    return eigen_vector_4x_(idx);
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    return data_[idx];
#endif
}

template <typename Scalar>
const Scalar& Vector<Scalar,4>::operator[] (unsigned int idx) const
{
    if(idx>=4)
    {
        std::cout<<"Vector index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    return eigen_vector_4x_(idx);
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    return data_[idx];
#endif
}

template <typename Scalar>
Vector<Scalar,4> Vector<Scalar,4>::operator+ (const Vector<Scalar,4> &vec4) const
{
    Scalar result[4];
    for(int i = 0; i < 4; ++i)
        result[i] = (*this)[i] + vec4[i];
    return Vector<Scalar,4>(result[0], result[1], result[2], result[3]);
}

template <typename Scalar>
Vector<Scalar,4>& Vector<Scalar,4>::operator+= (const Vector<Scalar,4> &vec4)
{
    for(int i = 0; i < 4; ++i)
        (*this)[i] = (*this)[i] + vec4[i];
    return *this;
}

template <typename Scalar>
Vector<Scalar,4> Vector<Scalar,4>::operator- (const Vector<Scalar,4> &vec4) const
{
    Scalar result[4];
    for(int i = 0; i < 4; ++i)
        result[i] = (*this)[i] - vec4[i];
    return Vector<Scalar,4>(result[0],result[1],result[2],result[3]);
}

template <typename Scalar>
Vector<Scalar,4>& Vector<Scalar,4>::operator-= (const Vector<Scalar,4> &vec4)
{
    for(int i = 0; i < 4; ++i)
        (*this)[i] = (*this)[i] - vec4[i];
    return *this;
}

template <typename Scalar>
Vector<Scalar,4>& Vector<Scalar,4>::operator= (const Vector<Scalar,4> &vec4)
{
    for(int i = 0; i < 4; ++i)
        (*this)[i] = vec4[i];
    return *this;
}

template <typename Scalar>
bool Vector<Scalar,4>::operator== (const Vector<Scalar,4> &vec4) const
{
    for(int i = 0; i < 4; ++i)
        if((*this)[i] != vec4[i])
            return false;
    return true;
}

template <typename Scalar>
bool Vector<Scalar,4>::operator!= (const Vector<Scalar,4> &vec2) const
{
    return !((*this)==vec2);
}

template <typename Scalar>
Vector<Scalar,4> Vector<Scalar,4>::operator* (Scalar scale) const
{
    Scalar result[4];
    for(int i = 0; i < 4; ++i)
        result[i] = (*this)[i] * scale;
    return Vector<Scalar,4>(result[0],result[1],result[2],result[3]);
}


template <typename Scalar>
Vector<Scalar, 4> Vector<Scalar, 4>::operator-(Scalar value) const
{
    Scalar result[4];
    for(unsigned int i = 0; i < 4; ++i)
        result[i] = (*this)[i] - value;
    return  Vector<Scalar,4>(result[0],result[1],result[2], result[3]);
}

template <typename Scalar>
Vector<Scalar, 4> Vector<Scalar, 4>::operator+(Scalar value) const
{
    Scalar result[4];
    for(unsigned int i = 0; i < 4; ++i)
        result[i] = (*this)[i] + value;
    return  Vector<Scalar,4>(result[0],result[1],result[2],result[3]);
}


template <typename Scalar>
Vector<Scalar,4>& Vector<Scalar,4>::operator+= (Scalar value)
{
    for(unsigned int i = 0; i < 4; ++i)
        (*this)[i] = (*this)[i] + value;
    return *this;
}

template <typename Scalar>
Vector<Scalar,4>& Vector<Scalar,4>::operator-= (Scalar value)
{
    for(unsigned int i = 0; i < 4; ++i)
        (*this)[i] = (*this)[i] - value;
    return *this;
}

template <typename Scalar>
Vector<Scalar,4>& Vector<Scalar,4>::operator*= (Scalar scale)
{
    for(int i = 0; i < 4; ++i)
        (*this)[i] = (*this)[i] * scale;
    return *this;
}

template <typename Scalar>
Vector<Scalar,4> Vector<Scalar,4>::operator/ (Scalar scale) const
{
    if(abs(scale)<std::numeric_limits<Scalar>::epsilon())
    {
        std::cerr<<"Vector Divide by zero error!\n";
        std::exit(EXIT_FAILURE);
    }
    Scalar result[4];
    for(int i = 0; i < 4; ++i)
        result[i] = (*this)[i] / scale;
    return Vector<Scalar,4>(result[0],result[1],result[2],result[3]);
}

template <typename Scalar>
Vector<Scalar,4>& Vector<Scalar,4>::operator/= (Scalar scale)
{
    if(abs(scale)<std::numeric_limits<Scalar>::epsilon())
    {
        std::cerr<<"Vector Divide by zero error!\n";
        std::exit(EXIT_FAILURE);
    }
    for(int i = 0; i < 4; ++i)
        (*this)[i] = (*this)[i] / scale;
    return *this;
}

template <typename Scalar>
Scalar Vector<Scalar,4>::norm() const
{
    Scalar result = (*this)[0]*(*this)[0] + (*this)[1]*(*this)[1]+(*this)[2]*(*this)[2] + (*this)[3]*(*this)[3];
    result = static_cast<Scalar>(sqrt(result));
    return result;
}

template <typename Scalar>
Scalar Vector<Scalar,4>::normSquared() const
{
    Scalar result = (*this)[0]*(*this)[0] + (*this)[1]*(*this)[1]+(*this)[2]*(*this)[2] + (*this)[3]*(*this)[3];
    return result;
}

template <typename Scalar>
Vector<Scalar,4>& Vector<Scalar,4>::normalize()
{
    Scalar norm = (*this).norm();
    bool nonzero_norm = norm > std::numeric_limits<Scalar>::epsilon();
    if(nonzero_norm)
    {
        for(int i = 0; i < 4; ++i)
        (*this)[i] = (*this)[i] / norm;
    }
    return *this;
}

template <typename Scalar>
Vector<Scalar,4> Vector<Scalar,4>::operator-(void) const
{
    return Vector<Scalar,4>(-(*this)[0],-(*this)[1],-(*this)[2],-(*this)[3]);
}

template <typename Scalar>
Scalar Vector<Scalar,4>::dot(const Vector<Scalar,4>& vec4) const
{
    return (*this)[0]*vec4[0] + (*this)[1]*vec4[1] + (*this)[2]*vec4[2] + (*this)[3]*vec4[3];
}

template <typename Scalar>
SquareMatrix<Scalar,4> Vector<Scalar,4>::outerProduct(const Vector<Scalar,4> &vec4) const
{
    SquareMatrix<Scalar,4> result;
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j < 4; ++j)
            result(i,j) = (*this)[i]*vec4[j];
    return result;
}

//explicit instantiation of template so that it could be compiled into a lib
template class Vector<unsigned char,4>;
template class Vector<unsigned short,4>;
template class Vector<unsigned int,4>;
template class Vector<unsigned long,4>;
template class Vector<unsigned long long,4>;
template class Vector<signed char,4>;
template class Vector<short,4>;
template class Vector<int,4>;
template class Vector<long,4>;
template class Vector<long long,4>;
template class Vector<float,4>;
template class Vector<double,4>;
template class Vector<long double,4>;

} //end of namespace Physika
