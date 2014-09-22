/*
 * @file vector_1d.cpp 
 * @brief 1d vector.
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
#include <cstdlib>
#include <iostream>
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Matrices/matrix_1x1.h"
#include "Physika_Core/Vectors/vector_1d.h"

namespace Physika{

template <typename Scalar>
Vector<Scalar,1>::Vector()
{
}

template <typename Scalar>
Vector<Scalar,1>::Vector(Scalar x)
{
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    eigen_vector_1x_(0)=x;
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    data_=x;
#endif
}

template <typename Scalar>
Vector<Scalar,1>::Vector(const Vector<Scalar,1> &vec2)
{
    *this = vec2;
}

template <typename Scalar>
Vector<Scalar,1>::~Vector()
{
}

template <typename Scalar>
Scalar& Vector<Scalar,1>::operator[] (unsigned int idx)
{
    if(idx>=1)
    {
        std::cout<<"Vector index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    return eigen_vector_1x_(idx);
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    return data_;
#endif
}

template <typename Scalar>
const Scalar& Vector<Scalar,1>::operator[] (unsigned int idx) const
{
    if(idx>=1)
    {
        std::cout<<"Vector index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    return eigen_vector_1x_(idx);
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    return data_;
#endif
}

template <typename Scalar>
Vector<Scalar,1> Vector<Scalar,1>::operator+ (const Vector<Scalar,1> &vec2) const
{
    Scalar result;
    result = (*this)[0] + vec2[0];
    return Vector<Scalar,1>(result);
}

template <typename Scalar>
Vector<Scalar,1>& Vector<Scalar,1>::operator+= (const Vector<Scalar,1> &vec2)
{
    (*this)[0] = (*this)[0] + vec2[0];
    return *this;
}

template <typename Scalar>
Vector<Scalar,1> Vector<Scalar,1>::operator- (const Vector<Scalar,1> &vec2) const
{
    Scalar result;
    result = (*this)[0] - vec2[0];
    return Vector<Scalar,1>(result);
}

template <typename Scalar>
Vector<Scalar,1>& Vector<Scalar,1>::operator-= (const Vector<Scalar,1> &vec2)
{
    (*this)[0] = (*this)[0] - vec2[0];
    return *this;
}

template <typename Scalar>
Vector<Scalar,1>& Vector<Scalar,1>::operator= (const Vector<Scalar,1> &vec2)
{
    (*this)[0] = vec2[0];
    return *this;
}

template <typename Scalar>
bool Vector<Scalar,1>::operator== (const Vector<Scalar,1> &vec2) const
{
    if((*this)[0] != vec2[0])
        return false;
    return true;
}

template <typename Scalar>
bool Vector<Scalar,1>::operator!= (const Vector<Scalar,1> &vec2) const
{
    return !((*this)==vec2);
}

template <typename Scalar>
Vector<Scalar,1> Vector<Scalar,1>::operator* (Scalar scale) const
{
    Scalar result;
    result = (*this)[0] * scale;
    return Vector<Scalar,1>(result);
}



template <typename Scalar>
Vector<Scalar,1> Vector<Scalar,1>::operator-(Scalar value) const
{
    Scalar result;
    result = (*this)[0] - value;
    return  Vector<Scalar,1>(result);
}

template <typename Scalar>
Vector<Scalar,1> Vector<Scalar,1>::operator+(Scalar value) const
{
    Scalar result;
    result = (*this)[0] + value;
    return  Vector<Scalar,1>(result);
}


template <typename Scalar>
Vector<Scalar,1>& Vector<Scalar,1>::operator+= (Scalar value)
{
    (*this)[0] = (*this)[0] + value;
    return *this;
}

template <typename Scalar>
Vector<Scalar,1>& Vector<Scalar,1>::operator-= (Scalar value)
{
    (*this)[0] = (*this)[0] - value;
    return *this;
}

template <typename Scalar>
Vector<Scalar,1>& Vector<Scalar,1>::operator*= (Scalar scale)
{
    (*this)[0] = (*this)[0] * scale;
    return *this;
}

template <typename Scalar>
Vector<Scalar,1> Vector<Scalar,1>::operator/ (Scalar scale) const
{
    if(abs(scale)<std::numeric_limits<Scalar>::epsilon())
    {
        std::cerr<<"Vector Divide by zero error!\n";
        std::exit(EXIT_FAILURE);
    }
    Scalar result;
    result = (*this)[0] / scale;
    return Vector<Scalar,1>(result);
}

template <typename Scalar>
Vector<Scalar,1>& Vector<Scalar,1>::operator/= (Scalar scale)
{
    if(abs(scale)<std::numeric_limits<Scalar>::epsilon())
    {
        std::cerr<<"Vector Divide by zero error!\n";
        std::exit(EXIT_FAILURE);
    }
    (*this)[0] = (*this)[0] / scale;
    return *this;
}

template <typename Scalar>
Scalar Vector<Scalar,1>::norm() const
{
    return (*this)[0];
}

template <typename Scalar>
Scalar Vector<Scalar,1>::normSquared() const
{
    Scalar result = (*this)[0]*(*this)[0];
    return result;
}

template <typename Scalar>
Vector<Scalar,1>& Vector<Scalar,1>::normalize()
{
    Scalar norm = (*this).norm();
    bool nonzero_norm = norm > std::numeric_limits<Scalar>::epsilon();
    if(nonzero_norm)
        (*this)[0] = (*this)[0] / norm;
    return *this;
}

template <typename Scalar>
Scalar Vector<Scalar,1>::cross(const Vector<Scalar,1>& vec2) const
{
    //cross product of 2 1d vectors is 0
    return 0;
}

template <typename Scalar>
Vector<Scalar,1> Vector<Scalar,1>::operator-(void) const
{
    return Vector<Scalar,1>(-(*this)[0]);
}

template <typename Scalar>
Scalar Vector<Scalar,1>::dot(const Vector<Scalar,1>& vec2) const
{
    return (*this)[0]*vec2[0];
}

template <typename Scalar>
SquareMatrix<Scalar,1> Vector<Scalar,1>::outerProduct(const Vector<Scalar,1> &vec2) const
{
    SquareMatrix<Scalar,1> result;
    for(unsigned int i = 0; i < 1; ++i)
        for(unsigned int j = 0; j < 1; ++j)
            result(i,j) = (*this)[i]*vec2[j];
    return result;
}

//explicit instantiation of template so that it could be compiled into a lib
template class Vector<unsigned char,1>;
template class Vector<unsigned short,1>;
template class Vector<unsigned int,1>;
template class Vector<unsigned long,1>;
template class Vector<unsigned long long,1>;
template class Vector<signed char,1>;
template class Vector<short,1>;
template class Vector<int,1>;
template class Vector<long,1>;
template class Vector<long long,1>;
template class Vector<float,1>;
template class Vector<double,1>;
template class Vector<long double,1>;

} //end of namespace Physika
