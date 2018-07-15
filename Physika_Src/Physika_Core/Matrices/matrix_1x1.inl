/*
 * @file matrix_1x1.cpp
 * @brief 1x1 matrix.
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
#include <glm/gtc/epsilon.hpp>

//#include "Physika_Core/Utilities/physika_exception.h"
//#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Vectors/vector_1d.h"
//#include "Physika_Core/Matrices/matrix_1x1.h"

namespace Physika{

template <typename Scalar>
COMM_FUNC SquareMatrix<Scalar,1>::SquareMatrix()
    :SquareMatrix<Scalar, 1>(0) //delegating ctor
{
}

template <typename Scalar>
COMM_FUNC SquareMatrix<Scalar,1>::SquareMatrix(Scalar value)
    :data_(value)
{
}

template <typename Scalar>
COMM_FUNC Scalar& SquareMatrix<Scalar,1>::operator() (unsigned int i, unsigned int j)
{
    return const_cast<Scalar &>(static_cast<const SquareMatrix<Scalar, 1> &>(*this)(i, j));
}

template <typename Scalar>
COMM_FUNC const Scalar& SquareMatrix<Scalar,1>::operator() (unsigned int i, unsigned int j) const
{
// #ifndef __CUDA_ARCH__
//     bool index_valid = (i < 1) && (j < 1);
//     if(!index_valid)
//         throw PhysikaException("Matrix index out of range!");
// #endif
    return data_;
}

template <typename Scalar>
COMM_FUNC const Vector<Scalar,1> SquareMatrix<Scalar,1>::rowVector(unsigned int i) const
{
    Scalar val = (*this)(i,i);
    return Vector<Scalar,1>(val);
}

template <typename Scalar>
COMM_FUNC const Vector<Scalar,1> SquareMatrix<Scalar,1>::colVector(unsigned int i) const
{
    Scalar val = (*this)(i,i);
    return Vector<Scalar,1>(val);
}

template <typename Scalar>
COMM_FUNC const SquareMatrix<Scalar,1> SquareMatrix<Scalar,1>::operator+ (const SquareMatrix<Scalar,1> &mat2) const
{
    return SquareMatrix<Scalar,1>(*this) += mat2;
}

template <typename Scalar>
COMM_FUNC SquareMatrix<Scalar,1>& SquareMatrix<Scalar,1>::operator+= (const SquareMatrix<Scalar,1> &mat2)
{
    data_ += mat2.data_;
    return *this;
}

template <typename Scalar>
COMM_FUNC const SquareMatrix<Scalar,1> SquareMatrix<Scalar,1>::operator- (const SquareMatrix<Scalar,1> &mat2) const
{
    return SquareMatrix<Scalar, 1>(*this) -= mat2;
}

template <typename Scalar>
COMM_FUNC SquareMatrix<Scalar,1>& SquareMatrix<Scalar,1>::operator-= (const SquareMatrix<Scalar,1> &mat2)
{
    data_ -= mat2.data_;
    return *this;
}

template <typename Scalar>
COMM_FUNC bool SquareMatrix<Scalar,1>::operator== (const SquareMatrix<Scalar,1> &mat2) const
{
    return glm::abs(data_ - mat2.data_) < 1e-6;
}

template <typename Scalar>
COMM_FUNC bool SquareMatrix<Scalar,1>::operator!= (const SquareMatrix<Scalar,1> &mat2) const
{
    return !((*this)==mat2);
}

template <typename Scalar>
COMM_FUNC const SquareMatrix<Scalar,1> SquareMatrix<Scalar,1>::operator* (Scalar scale) const
{
    return SquareMatrix<Scalar, 1>(*this) *= scale;
}

template <typename Scalar>
COMM_FUNC SquareMatrix<Scalar,1>& SquareMatrix<Scalar,1>::operator*= (Scalar scale)
{
    data_ *= scale;
    return *this;
}

template <typename Scalar>
COMM_FUNC const Vector<Scalar,1> SquareMatrix<Scalar,1>::operator* (const Vector<Scalar,1> &vec) const
{
    Vector<Scalar,1> result(0);
    result[0] += (*this)(0, 0) * vec[0];
    return result;
}

template <typename Scalar>
COMM_FUNC const SquareMatrix<Scalar,1> SquareMatrix<Scalar,1>::operator* (const SquareMatrix<Scalar,1> &mat2) const
{
    return SquareMatrix<Scalar, 1>(*this) *= mat2;
}

template <typename Scalar>
COMM_FUNC SquareMatrix<Scalar, 1> & SquareMatrix<Scalar, 1>::operator*= (const SquareMatrix<Scalar, 1> &mat2)
{
    data_ *= mat2.data_;
    return *this;
}


template <typename Scalar>
COMM_FUNC const SquareMatrix<Scalar,1> SquareMatrix<Scalar,1>::operator/ (Scalar scale) const
{
    return SquareMatrix<Scalar,1>(*this) /= scale;
}

template <typename Scalar>
COMM_FUNC SquareMatrix<Scalar,1>& SquareMatrix<Scalar,1>::operator/= (Scalar scale)
{
// #ifndef __CUDA_ARCH__
//     if (abs(scale) <= std::numeric_limits<Scalar>::epsilon())
//         throw PhysikaException("Matrix Divide by zero error!");
// #endif
    data_ /= scale;
    return *this;
}

template <typename Scalar>
COMM_FUNC const SquareMatrix<Scalar, 1> SquareMatrix<Scalar, 1>::operator- (void) const
{
    SquareMatrix<Scalar, 1> res;
    res.data_ = -data_;
    return res;
}

template <typename Scalar>
COMM_FUNC const SquareMatrix<Scalar,1> SquareMatrix<Scalar,1>::transpose() const
{
    return SquareMatrix<Scalar,1>(*this);
}

template <typename Scalar>
COMM_FUNC const SquareMatrix<Scalar,1> SquareMatrix<Scalar,1>::inverse() const
{
    Scalar data = (*this)(0,0);
// #ifndef __CUDA_ARCH__
//     if (data == 0)
//         throw PhysikaException("Matrix not invertible!");
// #endif
    return SquareMatrix<Scalar,1>(1/data);
}

template <typename Scalar>
COMM_FUNC Scalar SquareMatrix<Scalar,1>::determinant() const
{
    return data_;
}

template <typename Scalar>
COMM_FUNC Scalar SquareMatrix<Scalar,1>::trace() const
{
    return data_;
}

template <typename Scalar>
COMM_FUNC Scalar SquareMatrix<Scalar,1>::doubleContraction(const SquareMatrix<Scalar,1> &mat2) const
{
    Scalar result = 0;
    result += (*this)(0, 0)*mat2(0, 0);
    return result;
}

template <typename Scalar>
COMM_FUNC Scalar SquareMatrix<Scalar,1>::frobeniusNorm() const
{
    return glm::abs(data_);
}

template <typename Scalar>
COMM_FUNC const SquareMatrix<Scalar,1> SquareMatrix<Scalar,1>::identityMatrix()
{
    return SquareMatrix<Scalar,1>(1);
}

template <typename S, typename T>
COMM_FUNC  const SquareMatrix<T, 1> operator* (S scale, const SquareMatrix<T, 1> &mat)
{
	return mat*scale;
}

//explicit instantiation of template so that it could be compiled into a lib
// template class SquareMatrix<unsigned char, 1>;
// template class SquareMatrix<unsigned short, 1>;
// template class SquareMatrix<unsigned int, 1>;
// template class SquareMatrix<unsigned long, 1>;
// template class SquareMatrix<unsigned long long, 1>;
// template class SquareMatrix<signed char, 1>;
// template class SquareMatrix<short, 1>;
// template class SquareMatrix<int, 1>;
// template class SquareMatrix<long, 1>;
// template class SquareMatrix<long long, 1>;
// template class SquareMatrix<float,1>;
// template class SquareMatrix<double,1>;
//template class SquareMatrix<long double,1>;

}  //end of namespace Physika
