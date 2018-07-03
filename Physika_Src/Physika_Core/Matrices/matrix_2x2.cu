/*
 * @file matrix_2x2.cu
 * @brief 2x2 matrix.
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

#include <cmath>
#include <limits>

//#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Matrices/matrix_2x2.h"

namespace Physika{

template <typename Scalar>
CPU_GPU_FUNC_DECL SquareMatrix<Scalar,2>::SquareMatrix()
    :SquareMatrix(0) //delegating ctor
{
}

template <typename Scalar>
CPU_GPU_FUNC_DECL SquareMatrix<Scalar,2>::SquareMatrix(Scalar value)
    :SquareMatrix(value, value, value, value) //delegating ctor
{
}

template <typename Scalar>
CPU_GPU_FUNC_DECL SquareMatrix<Scalar,2>::SquareMatrix(Scalar x00, Scalar x01, Scalar x10, Scalar x11)
    :data_(x00, x10, x01, x11)
{
}

template <typename Scalar>
CPU_GPU_FUNC_DECL SquareMatrix<Scalar,2>::SquareMatrix(const Vector<Scalar,2> &row1, const Vector<Scalar,2> &row2)
    :data_(row1[0], row2[0], row1[1], row2[1])
{
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Scalar& SquareMatrix<Scalar,2>::operator() (unsigned int i, unsigned int j)
{
    return const_cast<Scalar &>(static_cast<const SquareMatrix<Scalar, 2> &>(*this)(i, j));
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Scalar& SquareMatrix<Scalar,2>::operator() (unsigned int i, unsigned int j) const
{
// #ifndef __CUDA_ARCH__
//     bool index_valid = (i < 2) && (j < 2);
//     if (!index_valid)
//         throw PhysikaException("Matrix index out of range!");
// #endif
    //Note: glm is column-based
    return data_[j][i];
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Vector<Scalar,2> SquareMatrix<Scalar,2>::rowVector(unsigned int i) const
{
// #ifndef __CUDA_ARCH__
//     if(i>=2)
//         throw PhysikaException("Matrix index out of range!");
// #endif
    Vector<Scalar,2> result((*this)(i,0),(*this)(i,1));
    return result;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Vector<Scalar,2> SquareMatrix<Scalar,2>::colVector(unsigned int i) const
{
// #ifndef __CUDA_ARCH__
//     if(i>=2)
//         throw PhysikaException("Matrix index out of range!");
// #endif
    Vector<Scalar,2> result((*this)(0,i),(*this)(1,i));
    return result;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,2> SquareMatrix<Scalar,2>::operator+ (const SquareMatrix<Scalar,2> &mat2) const
{
    return SquareMatrix<Scalar, 2>(*this) += mat2;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL SquareMatrix<Scalar,2>& SquareMatrix<Scalar,2>::operator+= (const SquareMatrix<Scalar,2> &mat2)
{
    data_ += mat2.data_;
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,2> SquareMatrix<Scalar,2>::operator- (const SquareMatrix<Scalar,2> &mat2) const
{
    return SquareMatrix<Scalar, 2>(*this) -= mat2;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL SquareMatrix<Scalar,2>& SquareMatrix<Scalar,2>::operator-= (const SquareMatrix<Scalar,2> &mat2)
{
    data_ -= mat2.data_;
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL bool SquareMatrix<Scalar,2>::operator== (const SquareMatrix<Scalar,2> &mat2) const
{
    return data_ == mat2.data_;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL bool SquareMatrix<Scalar,2>::operator!= (const SquareMatrix<Scalar,2> &mat2) const
{
    return !((*this)==mat2);
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,2> SquareMatrix<Scalar,2>::operator* (Scalar scale) const
{
    return SquareMatrix<Scalar, 2>(*this) *= scale;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL SquareMatrix<Scalar,2>& SquareMatrix<Scalar,2>::operator*= (Scalar scale)
{
    data_ *= scale;
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Vector<Scalar,2> SquareMatrix<Scalar,2>::operator* (const Vector<Scalar,2> &vec) const
{
    Vector<Scalar,2> result(0);
    for (unsigned int i = 0; i < 2; ++i)
        for (unsigned int j = 0; j < 2; ++j)
            result[i] += (*this)(i, j) * vec[j];
    return result;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,2> SquareMatrix<Scalar,2>::operator* (const SquareMatrix<Scalar,2> &mat2) const
{
    return SquareMatrix<Scalar, 2>(*this) *= mat2;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL SquareMatrix<Scalar,2>& SquareMatrix<Scalar,2>::operator*= (const SquareMatrix<Scalar,2> &mat2)
{
    data_ *= mat2.data_;
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,2> SquareMatrix<Scalar,2>::operator/ (Scalar scale) const
{
    return SquareMatrix<Scalar, 2>(*this) /= scale;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL SquareMatrix<Scalar,2>& SquareMatrix<Scalar,2>::operator/= (Scalar scale)
{
// #ifndef __CUDA_ARCH__
//     if (abs(scale) <= std::numeric_limits<Scalar>::epsilon())
//         throw PhysikaException("Matrix Divide by zero error!");
// #endif
    data_ /= scale;
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const SquareMatrix<Scalar, 2> SquareMatrix<Scalar, 2>::operator- (void) const
{
    SquareMatrix<Scalar, 2> res;
    res.data_ = -data_;
    return res;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,2> SquareMatrix<Scalar,2>::transpose() const
{
    SquareMatrix<Scalar, 2> res;
    res.data_ = glm::transpose(data_);
    return res;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,2> SquareMatrix<Scalar,2>::inverse() const
{
    SquareMatrix<Scalar, 2> res;
    res.data_ = glm::inverse(data_);

// #ifndef __CUDA_ARCH__
//     double fir_ele = static_cast<double>(res(0, 0));
//     if(std::isnan(fir_ele) == true || std::isinf(fir_ele) == true)
//         throw PhysikaException("Matrix not invertible!");
// #endif

    return res;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Scalar SquareMatrix<Scalar,2>::determinant() const
{
    return glm::determinant(data_);
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Scalar SquareMatrix<Scalar,2>::trace() const
{
    return (*this)(0,0) + (*this)(1,1);
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Scalar SquareMatrix<Scalar,2>::doubleContraction(const SquareMatrix<Scalar,2> &mat2) const
{
    Scalar result = 0;
    for (unsigned int i = 0; i < 2; ++i)
        for (unsigned int j = 0; j < 2; ++j)
            result += (*this)(i, j)*mat2(i, j);
    return result;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Scalar SquareMatrix<Scalar,2>::frobeniusNorm() const
{
    Scalar result = 0;
    for (unsigned int i = 0; i < 2; ++i)
        for (unsigned int j = 0; j < 2; ++j)
            result += (*this)(i, j)*(*this)(i, j);
    return glm::sqrt(result);
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,2> SquareMatrix<Scalar,2>::identityMatrix()
{
    return SquareMatrix<Scalar,2>(1.0,0.0,
                                  0.0,1.0);
}

//explicit instantiation of template so that it could be compiled into a lib
template class SquareMatrix<unsigned char, 2>;
template class SquareMatrix<unsigned short, 2>;
template class SquareMatrix<unsigned int, 2>;
template class SquareMatrix<unsigned long, 2>;
template class SquareMatrix<unsigned long long, 2>;
template class SquareMatrix<signed char, 2>;
template class SquareMatrix<short, 2>;
template class SquareMatrix<int, 2>;
template class SquareMatrix<long, 2>;
template class SquareMatrix<long long, 2>;
template class SquareMatrix<float,2>;
template class SquareMatrix<double,2>;
template class SquareMatrix<long double,2>;

}  //end of namespace Physika
