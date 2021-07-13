/*
 * @file vector_2d.cpp 
 * @brief 2d vector.
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
#include <glm/gtx/norm.hpp>
//#include "Physika_Core/Utilities/physika_assert.h"
//#include "Physika_Core/Utilities/math_utilities.h"
//#include "Physika_Core/Utilities/physika_exception.h"
//#include "Physika_Core/Matrices/matrix_2x2.h"

namespace PhysIKA {

template <typename Scalar>
COMM_FUNC Vector<Scalar, 2>::Vector()
    : Vector(0)  //delegating ctor
{
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 2>::Vector(Scalar x)
    : Vector(x, x)  //delegating ctor
{
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 2>::Vector(Scalar x, Scalar y)
    : data_(x, y)
{
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 2>::Vector(const Vector<Scalar, 2>& vec)
    : data_(vec.data_)
{
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 2>::~Vector()
{
}

template <typename Scalar>
COMM_FUNC Scalar& Vector<Scalar, 2>::operator[](unsigned int idx)
{
    return const_cast<Scalar&>(static_cast<const Vector<Scalar, 2>&>(*this)[idx]);
}

template <typename Scalar>
COMM_FUNC const Scalar& Vector<Scalar, 2>::operator[](unsigned int idx) const
{
    // #ifndef __CUDA_ARCH__
    //     if (idx >= 2)
    //         throw PhysikaException("Vector index out of range!");
    // #endif
    return data_[idx];
}

template <typename Scalar>
COMM_FUNC const Vector<Scalar, 2> Vector<Scalar, 2>::operator+(const Vector<Scalar, 2>& vec2) const
{
    return Vector<Scalar, 2>(*this) += vec2;
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 2>& Vector<Scalar, 2>::operator+=(const Vector<Scalar, 2>& vec2)
{
    data_ += vec2.data_;
    return *this;
}

template <typename Scalar>
COMM_FUNC const Vector<Scalar, 2> Vector<Scalar, 2>::operator-(const Vector<Scalar, 2>& vec2) const
{
    return Vector<Scalar, 2>(*this) -= vec2;
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 2>& Vector<Scalar, 2>::operator-=(const Vector<Scalar, 2>& vec2)
{
    data_ -= vec2.data_;
    return *this;
}

template <typename Scalar>
COMM_FUNC const Vector<Scalar, 2> Vector<Scalar, 2>::operator*(const Vector<Scalar, 2>& vec2) const
{
    return Vector<Scalar, 2>(data_[0] * vec2[0], data_[1] * vec2[1]);
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 2>& Vector<Scalar, 2>::operator*=(const Vector<Scalar, 2>& vec2)
{
    data_[0] *= vec2.data_[0];
    data_[1] *= vec2.data_[1];
    return *this;
}

template <typename Scalar>
COMM_FUNC const Vector<Scalar, 2> Vector<Scalar, 2>::operator/(const Vector<Scalar, 2>& vec2) const
{
    return Vector<Scalar, 2>(data_[0] / vec2[0], data_[1] / vec2[1]);
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 2>& Vector<Scalar, 2>::operator/=(const Vector<Scalar, 2>& vec2)
{
    data_[0] /= vec2.data_[0];
    data_[1] /= vec2.data_[1];
    ;
    return *this;
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 2>& Vector<Scalar, 2>::operator=(const Vector<Scalar, 2>& vec2)
{
    data_ = vec2.data_;
    return *this;
}

template <typename Scalar>
COMM_FUNC bool Vector<Scalar, 2>::operator==(const Vector<Scalar, 2>& vec2) const
{
    return data_ == vec2.data_;
}

template <typename Scalar>
COMM_FUNC bool Vector<Scalar, 2>::operator!=(const Vector<Scalar, 2>& vec2) const
{
    return !((*this) == vec2);
}

template <typename Scalar>
COMM_FUNC const Vector<Scalar, 2> Vector<Scalar, 2>::operator+(Scalar value) const
{
    return Vector<Scalar, 2>(*this) += value;
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 2>& Vector<Scalar, 2>::operator+=(Scalar value)
{
    data_ += value;
    return *this;
}

template <typename Scalar>
COMM_FUNC const Vector<Scalar, 2> Vector<Scalar, 2>::operator-(Scalar value) const
{
    return Vector<Scalar, 2>(*this) -= value;
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 2>& Vector<Scalar, 2>::operator-=(Scalar value)
{
    data_ -= value;
    return *this;
}

template <typename Scalar>
COMM_FUNC const Vector<Scalar, 2> Vector<Scalar, 2>::operator*(Scalar scale) const
{
    return Vector<Scalar, 2>(*this) *= scale;
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 2>& Vector<Scalar, 2>::operator*=(Scalar scale)
{
    data_ *= scale;
    return *this;
}

template <typename Scalar>
COMM_FUNC const Vector<Scalar, 2> Vector<Scalar, 2>::operator/(Scalar scale) const
{
    return Vector<Scalar, 2>(*this) /= scale;
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 2>& Vector<Scalar, 2>::operator/=(Scalar scale)
{
    // #ifndef __CUDA_ARCH__
    //     if (abs(scale) <= std::numeric_limits<Scalar>::epsilon())
    //         throw PhysikaException("Vector Divide by zero error!");
    // #endif
    data_ /= scale;
    return *this;
}

template <typename Scalar>
COMM_FUNC const Vector<Scalar, 2> Vector<Scalar, 2>::operator-(void) const
{
    Vector<Scalar, 2> res;
    res.data_ = -data_;
    return res;
}

template <typename Scalar>
COMM_FUNC Scalar Vector<Scalar, 2>::norm() const
{
    return glm::length(data_);
}

template <typename Scalar>
COMM_FUNC Scalar Vector<Scalar, 2>::normSquared() const
{
    return glm::length2(data_);
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 2>& Vector<Scalar, 2>::normalize()
{
    data_ = glm::length(data_) > glm::epsilon<Scalar>() ? glm::normalize(data_) : glm::tvec2<Scalar>(0, 0);
    return *this;
}

template <typename Scalar>
COMM_FUNC Scalar Vector<Scalar, 2>::cross(const Vector<Scalar, 2>& vec2) const
{
    return (*this)[0] * vec2[1] - (*this)[1] * vec2[0];
}

template <typename Scalar>
COMM_FUNC Scalar Vector<Scalar, 2>::dot(const Vector<Scalar, 2>& vec2) const
{
    return glm::dot(data_, vec2.data_);
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 2> Vector<Scalar, 2>::minimum(const Vector<Scalar, 2>& vec2) const
{
    Vector<Scalar, 2> res;
    res[0] = data_[0] < vec2[0] ? data_[0] : vec2[0];
    res[1] = data_[1] < vec2[1] ? data_[1] : vec2[1];
    return res;
}

template <typename Scalar>
COMM_FUNC Vector<Scalar, 2> Vector<Scalar, 2>::maximum(const Vector<Scalar, 2>& vec2) const
{
    Vector<Scalar, 2> res;
    res[0] = data_[0] > vec2[0] ? data_[0] : vec2[0];
    res[1] = data_[1] > vec2[1] ? data_[1] : vec2[1];
    return res;
}

// template <typename Scalar>
// COMM_FUNC const SquareMatrix<Scalar,2> Vector<Scalar,2>::outerProduct(const Vector<Scalar,2> &vec2) const
// {
//     SquareMatrix<Scalar,2> result;
//     for(unsigned int i = 0; i < 2; ++i)
//         for(unsigned int j = 0; j < 2; ++j)
//             result(i,j) = (*this)[i]*vec2[j];
//     return result;
// }

//make * operator commutative
template <typename S, typename T>
COMM_FUNC const Vector<T, 2> operator*(S scale, const Vector<T, 2>& vec)
{
    return vec * ( T )scale;
}

//explicit instantiation of template so that it could be compiled into a lib
// template class Vector<unsigned char, 2>;
// template class Vector<unsigned short, 2>;
// template class Vector<unsigned int, 2>;
// template class Vector<unsigned long, 2>;
// template class Vector<unsigned long long, 2>;
// template class Vector<signed char, 2>;
// template class Vector<short, 2>;
// template class Vector<int, 2>;
// template class Vector<long, 2>;
// template class Vector<long long, 2>;
// template class Vector<float,2>;
// template class Vector<double,2>;
//template class Vector<long double,2>;

}  // namespace PhysIKA
