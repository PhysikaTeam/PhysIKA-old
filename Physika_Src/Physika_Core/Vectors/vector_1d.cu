/*
 * @file vector_1d.cpp 
 * @brief 1d vector.
 * @author Fei Zhu, Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <limits>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/epsilon.hpp>

#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Matrices/matrix_1x1.h"
#include "Physika_Core/Vectors/vector_1d.h"

namespace Physika{

template <typename Scalar>
CPU_GPU_FUNC_DECL Vector<Scalar,1>::Vector()
    :Vector(0) //delegating ctor
{
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Vector<Scalar,1>::Vector(Scalar x)
    :data_(x)
{
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Scalar& Vector<Scalar,1>::operator[] (unsigned int idx)
{
    return const_cast<Scalar &> (static_cast<const Vector<Scalar, 1> &>(*this)[idx]);
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Scalar& Vector<Scalar,1>::operator[] (unsigned int idx) const
{
#ifndef __CUDA_ARCH__
    if(idx>=1)
        throw PhysikaException("Vector index out of range!");
#endif
    return data_;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Vector<Scalar,1> Vector<Scalar,1>::operator+ (const Vector<Scalar,1> &vec2) const
{
    return Vector<Scalar,1>(*this) += vec2;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Vector<Scalar,1>& Vector<Scalar,1>::operator+= (const Vector<Scalar,1> &vec2)
{
    data_ += vec2.data_;
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Vector<Scalar,1> Vector<Scalar,1>::operator- (const Vector<Scalar,1> &vec2) const
{
    return Vector<Scalar, 1>(*this) -= vec2;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Vector<Scalar,1>& Vector<Scalar,1>::operator-= (const Vector<Scalar,1> &vec2)
{
    data_ -= vec2.data_;
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL bool Vector<Scalar,1>::operator== (const Vector<Scalar,1> &vec2) const
{
    return glm::abs(data_ - vec2.data_) < 1e-6;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL bool Vector<Scalar,1>::operator!= (const Vector<Scalar,1> &vec2) const
{
    return !((*this)==vec2);
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Vector<Scalar, 1> Vector<Scalar, 1>::operator+(Scalar value) const
{
    return Vector<Scalar, 1>(*this) += value;
}


template <typename Scalar>
CPU_GPU_FUNC_DECL Vector<Scalar, 1>& Vector<Scalar, 1>::operator+= (Scalar value)
{
    data_ += value;
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Vector<Scalar, 1> Vector<Scalar, 1>::operator-(Scalar value) const
{
    return Vector<Scalar, 1>(*this) -= value;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Vector<Scalar, 1>& Vector<Scalar, 1>::operator-= (Scalar value)
{
    data_ -= value;
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Vector<Scalar,1> Vector<Scalar,1>::operator* (Scalar scale) const
{
    return Vector<Scalar,1>(*this) *= scale;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Vector<Scalar, 1>& Vector<Scalar, 1>::operator*= (Scalar scale)
{
    data_ *= scale;
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Vector<Scalar,1> Vector<Scalar,1>::operator/ (Scalar scale) const
{
    return Vector<Scalar,1>(*this) /= scale;
}

    
template <typename Scalar>
CPU_GPU_FUNC_DECL Vector<Scalar,1>& Vector<Scalar,1>::operator/= (Scalar scale)
{
#ifndef __CUDA_ARCH__
    if(abs(scale) <= std::numeric_limits<Scalar>::epsilon())
        throw PhysikaException("Vector Divide by zero error!");
#endif

    data_ /= scale;
    return *this;
}
    

template <typename Scalar>
CPU_GPU_FUNC_DECL const Vector<Scalar, 1> Vector<Scalar, 1>::operator-(void) const
{
    return Vector<Scalar, 1>(-data_);
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Scalar Vector<Scalar,1>::norm() const
{
    return glm::abs(data_);
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Scalar Vector<Scalar,1>::normSquared() const
{
    return data_*data_;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Vector<Scalar,1>& Vector<Scalar,1>::normalize()
{
    Scalar norm = (*this).norm();
    bool nonzero_norm = norm > 1e-6;  //need further consideration
    if(nonzero_norm)
        (*this)[0] /= norm;
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Scalar Vector<Scalar,1>::cross(const Vector<Scalar,1>& vec2) const
{
    //cross product of 2 1d vectors is 0
    return 0;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Scalar Vector<Scalar,1>::dot(const Vector<Scalar,1>& vec2) const
{
    return data_ * vec2.data_;;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,1> Vector<Scalar,1>::outerProduct(const Vector<Scalar,1> &vec2) const
{
    SquareMatrix<Scalar,1> result;
    result(0, 0) = data_ * vec2.data_;
    return result;
}

//explicit instantiation of template so that it could be compiled into a lib
template class Vector<unsigned char, 1>;
template class Vector<unsigned short, 1>;
template class Vector<unsigned int, 1>;
template class Vector<unsigned long, 1>;
template class Vector<unsigned long long, 1>;
template class Vector<signed char, 1>;
template class Vector<short, 1>;
template class Vector<int, 1>;
template class Vector<long, 1>;
template class Vector<long long, 1>;
template class Vector<float,1>;
template class Vector<double,1>;
template class Vector<long double,1>;

} //end of namespace Physika
