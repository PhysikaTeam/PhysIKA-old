/*
 * @file vector_4d.cpp 
 * @brief 4d vector.
 * @author Liyou Xu, Fei Zhu, Wei Chen
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
//#include "Physika_Core/Utilities/physika_assert.h"
//#include "Physika_Core/Utilities/math_utilities.h"
//#include "Physika_Core/Utilities/physika_exception.h"
//#include "Physika_Core/Matrices/matrix_4x4.h"
//#include "Physika_Core/Vectors/vector_4d.h"

namespace PhysIKA {

template <typename Scalar>
COMM_FUNC Vector<Scalar, 4>::Vector()
    : Vector(0)  //delegating ctor
{
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 4>::Vector(Scalar x)
    : Vector(x, x, x, x)  //delegating ctor
{
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 4>::Vector(Scalar x, Scalar y, Scalar z, Scalar w)
    : data_(x, y, z, w)
{
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 4>::Vector(const Vector<Scalar, 4>& vec2)
    : data_(vec2.data_)
{
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 4>::~Vector()
{
}

template <typename Scalar>
COMM_FUNC Scalar& Vector<Scalar, 4>::operator[](unsigned int idx)
{
    return const_cast<Scalar&>(static_cast<const Vector<Scalar, 4>&>(*this)[idx]);
}

template <typename Scalar>
COMM_FUNC const Scalar& Vector<Scalar, 4>::operator[](unsigned int idx) const
{
    // #ifndef __CUDA_ARCH__
    //     if(idx>=4)
    //         throw PhysikaException("Vector index out of range!");
    // #endif
    return data_[idx];
}

template <typename Scalar>
COMM_FUNC const Vector<Scalar, 4> Vector<Scalar, 4>::operator+(const Vector<Scalar, 4>& vec2) const
{
    return Vector<Scalar, 4>(*this) += vec2;
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 4>& Vector<Scalar, 4>::operator+=(const Vector<Scalar, 4>& vec2)
{
    data_ += vec2.data_;
    return *this;
}

template <typename Scalar>
COMM_FUNC const Vector<Scalar, 4> Vector<Scalar, 4>::operator-(const Vector<Scalar, 4>& vec2) const
{
    return Vector<Scalar, 4>(*this) -= vec2;
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 4>& Vector<Scalar, 4>::operator-=(const Vector<Scalar, 4>& vec2)
{
    data_ -= vec2.data_;
    return *this;
}

template <typename Scalar>
COMM_FUNC const Vector<Scalar, 4> Vector<Scalar, 4>::operator*(const Vector<Scalar, 4>& vec2) const
{
    return Vector<Scalar, 4>(data_[0] * vec2.data_[0], data_[1] * vec2.data_[1], data_[2] * vec2.data_[2], data_[3] * vec2.data_[3]);
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 4>& Vector<Scalar, 4>::operator*=(const Vector<Scalar, 4>& vec2)
{
    data_[0] *= vec2.data_[0];
    data_[1] *= vec2.data_[1];
    data_[2] *= vec2.data_[2];
    data_[3] *= vec2.data_[3];
    return *this;
}

template <typename Scalar>
COMM_FUNC const Vector<Scalar, 4> Vector<Scalar, 4>::operator/(const Vector<Scalar, 4>& vec2) const
{
    return Vector<Scalar, 4>(data_[0] / vec2[0], data_[1] / vec2[1], data_[2] / vec2[2], data_[3] / vec2[3]);
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 4>& Vector<Scalar, 4>::operator/=(const Vector<Scalar, 4>& vec2)
{
    data_[0] /= vec2.data_[0];
    data_[1] /= vec2.data_[1];
    data_[2] /= vec2.data_[2];
    data_[3] /= vec2.data_[3];
    return *this;
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 4>& Vector<Scalar, 4>::operator=(const Vector<Scalar, 4>& vec2)
{
    data_ = vec2.data_;
    return *this;
}

template <typename Scalar>
COMM_FUNC bool Vector<Scalar, 4>::operator==(const Vector<Scalar, 4>& vec2) const
{
    return data_ == vec2.data_;
}

template <typename Scalar>
COMM_FUNC bool Vector<Scalar, 4>::operator!=(const Vector<Scalar, 4>& vec2) const
{
    return !((*this) == vec2);
}

template <typename Scalar>
COMM_FUNC const Vector<Scalar, 4> Vector<Scalar, 4>::operator+(Scalar value) const
{
    return Vector<Scalar, 4>(*this) += value;
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 4>& Vector<Scalar, 4>::operator+=(Scalar value)
{
    data_ += value;
    return *this;
}

template <typename Scalar>
COMM_FUNC const Vector<Scalar, 4> Vector<Scalar, 4>::operator-(Scalar value) const
{
    return Vector<Scalar, 4>(*this) -= value;
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 4>& Vector<Scalar, 4>::operator-=(Scalar value)
{
    data_ -= value;
    return *this;
}

template <typename Scalar>
COMM_FUNC const Vector<Scalar, 4> Vector<Scalar, 4>::operator*(Scalar scale) const
{
    return Vector<Scalar, 4>(*this) *= scale;
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 4>& Vector<Scalar, 4>::operator*=(Scalar scale)
{
    data_ *= scale;
    return *this;
}

template <typename Scalar>
COMM_FUNC const Vector<Scalar, 4> Vector<Scalar, 4>::operator/(Scalar scale) const
{
    return Vector<Scalar, 4>(*this) /= scale;
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 4>& Vector<Scalar, 4>::operator/=(Scalar scale)
{
    // #ifndef __CUDA_ARCH__
    //     if (abs(scale) <= std::numeric_limits<Scalar>::epsilon())
    //         throw PhysikaException("Vector Divide by zero error!");
    // #endif
    data_ /= scale;
    return *this;
}

template <typename Scalar>
COMM_FUNC const Vector<Scalar, 4> Vector<Scalar, 4>::operator-(void) const
{
    Vector<Scalar, 4> res;
    res.data_ = -data_;
    return res;
}

template <typename Scalar>
COMM_FUNC Scalar Vector<Scalar, 4>::norm() const
{
    return glm::length(data_);
}

template <typename Scalar>
COMM_FUNC Scalar Vector<Scalar, 4>::normSquared() const
{
    return glm::length2(data_);
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 4>& Vector<Scalar, 4>::normalize()
{
    data_ = glm::length(data_) > glm::epsilon<Scalar>() ? glm::normalize(data_) : glm::tvec4<Scalar>(0, 0, 0, 0);
    return *this;
}

template <typename Scalar>
COMM_FUNC Scalar Vector<Scalar, 4>::dot(const Vector<Scalar, 4>& vec2) const
{
    return glm::dot(data_, vec2.data_);
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 4> Vector<Scalar, 4>::minimum(const Vector<Scalar, 4>& vec2) const
{
    Vector<Scalar, 4> res;
    res[0] = data_[0] < vec2[0] ? data_[0] : vec2[0];
    res[1] = data_[1] < vec2[1] ? data_[1] : vec2[1];
    res[2] = data_[2] < vec2[2] ? data_[2] : vec2[2];
    res[3] = data_[3] < vec2[3] ? data_[3] : vec2[3];
    return res;
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 4> Vector<Scalar, 4>::maximum(const Vector<Scalar, 4>& vec2) const
{
    Vector<Scalar, 4> res;
    res[0] = data_[0] > vec2[0] ? data_[0] : vec2[0];
    res[1] = data_[1] > vec2[1] ? data_[1] : vec2[1];
    res[2] = data_[2] > vec2[2] ? data_[2] : vec2[2];
    res[3] = data_[3] > vec2[3] ? data_[3] : vec2[3];
    return res;
}

template <typename S, typename T>
COMM_FUNC const Vector<T, 4> operator*(S scale, const Vector<T, 4>& vec)
{
    return vec * ( T )scale;
}

// template <typename Scalar>
// COMM_FUNC const SquareMatrix<Scalar,4> Vector<Scalar,4>::outerProduct(const Vector<Scalar,4> &vec2) const
// {
//     SquareMatrix<Scalar, 4> result;
//     for (unsigned int i = 0; i < 4; ++i)
//         for (unsigned int j = 0; j < 4; ++j)
//             result(i, j) = (*this)[i] * vec2[j];
//     return result;
// }

//explicit instantiation of template so that it could be compiled into a lib
// template class Vector<unsigned char, 4>;
// template class Vector<unsigned short, 4>;
// template class Vector<unsigned int, 4>;
// template class Vector<unsigned long, 4>;
// template class Vector<unsigned long long, 4>;
// template class Vector<signed char, 4>;
// template class Vector<short, 4>;
// template class Vector<int, 4>;
// template class Vector<long, 4>;
// template class Vector<long long, 4>;
// template class Vector<float,4>;
// template class Vector<double,4>;
//template class Vector<long double,4>;

}  // namespace PhysIKA
