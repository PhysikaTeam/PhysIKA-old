/*
 * @file vector_3d.cpp 
 * @brief 3d vector.
 * @author Sheng Yang, Fei Zhu, Wei Chen
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
#include <glm/gtx/norm.hpp>

#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

template <typename Scalar>
CPU_GPU_FUNC_DECL Vector<Scalar,3>::Vector()
    :Vector(0) //delegating ctor
{
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Vector<Scalar, 3>::Vector(Scalar x)
    :Vector(x, x, x) //delegating ctor
{
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Vector<Scalar,3>::Vector(Scalar x, Scalar y, Scalar z)
    :data_(x, y, z)
{
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Scalar& Vector<Scalar,3>::operator[] (unsigned int idx)
{
    return const_cast<Scalar &> (static_cast<const Vector<Scalar, 3> &>(*this)[idx]);
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Scalar& Vector<Scalar,3>::operator[] (unsigned int idx) const
{
#ifndef __CUDA_ARCH__
    if(idx>=3)
        throw PhysikaException("Vector index out of range!");
#endif
    return data_[idx];
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Vector<Scalar,3> Vector<Scalar,3>::operator+ (const Vector<Scalar,3> &vec2) const
{
    return Vector<Scalar, 3>(*this) += vec2;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Vector<Scalar,3>& Vector<Scalar,3>::operator+= (const Vector<Scalar,3> &vec2)
{
    data_ += vec2.data_;
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Vector<Scalar,3> Vector<Scalar,3>::operator- (const Vector<Scalar,3> &vec2) const
{
    return Vector<Scalar, 3>(*this) -= vec2;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Vector<Scalar,3>& Vector<Scalar,3>::operator-= (const Vector<Scalar,3> &vec2)
{
    data_ -= vec2.data_;
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL bool Vector<Scalar,3>::operator== (const Vector<Scalar,3> &vec2) const
{
    return data_ == vec2.data_;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL bool Vector<Scalar,3>::operator!= (const Vector<Scalar,3> &vec2) const
{
    return !((*this)==vec2);
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Vector<Scalar, 3> Vector<Scalar, 3>::operator+(Scalar value) const
{
    return Vector<Scalar, 3>(*this) += value;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Vector<Scalar, 3>& Vector<Scalar, 3>::operator+= (Scalar value)
{
    data_ += value;
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Vector<Scalar, 3> Vector<Scalar, 3>::operator-(Scalar value) const
{
    return Vector<Scalar, 3>(*this) -= value;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Vector<Scalar, 3>& Vector<Scalar, 3>::operator-= (Scalar value)
{
    data_ -= value;
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Vector<Scalar,3> Vector<Scalar,3>::operator* (Scalar scale) const
{
    return Vector<Scalar, 3>(*this) *= scale;
}

template <typename Scalar>
Vector<Scalar,3>& Vector<Scalar,3>::operator*= (Scalar scale)
{
    data_ *= scale;
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Vector<Scalar,3> Vector<Scalar,3>::operator/ (Scalar scale) const
{
    return Vector<Scalar, 3>(*this) /= scale;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Vector<Scalar,3>& Vector<Scalar,3>::operator/= (Scalar scale)
{
#ifndef __CUDA_ARCH__
    if (abs(scale) <= std::numeric_limits<Scalar>::epsilon())
        throw PhysikaException("Vector Divide by zero error!");
#endif
    data_ /= scale;
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Vector<Scalar, 3> Vector<Scalar, 3>::operator-(void) const
{
    Vector<Scalar, 3> res;
    res.data_ = -data_;
    return res;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Scalar Vector<Scalar,3>::norm() const
{
    return glm::length(data_);
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Scalar Vector<Scalar,3>::normSquared() const
{
    return glm::length2(data_);
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Vector<Scalar,3>& Vector<Scalar,3>::normalize()
{
    data_ = glm::normalize(data_);
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Vector<Scalar,3> Vector<Scalar,3>::cross(const Vector<Scalar,3>& vec2) const
{
    Vector<Scalar, 3> res;
    res.data_ = glm::cross(data_, vec2.data_);
    return res;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Scalar Vector<Scalar,3>::dot(const Vector<Scalar,3>& vec2) const
{
    return glm::dot(data_, vec2.data_);
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,3> Vector<Scalar,3>::outerProduct(const Vector<Scalar,3> &vec2) const
{
    SquareMatrix<Scalar, 3> result;
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            result(i, j) = (*this)[i] * vec2[j];
    return result;
}

//explicit instantiation of template so that it could be compiled into a lib
template class Vector<unsigned char, 3>;
template class Vector<unsigned short, 3>;
template class Vector<unsigned int, 3>;
template class Vector<unsigned long, 3>;
template class Vector<unsigned long long, 3>;
template class Vector<signed char, 3>;
template class Vector<short, 3>;
template class Vector<int, 3>;
template class Vector<long, 3>;
template class Vector<long long, 3>;
template class Vector<float,3>;
template class Vector<double,3>;
template class Vector<long double,3>;

} //end of namespace Physika
