/*
 * @file matrix_3x3.cu
 * @brief 3x3 matrix.
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

#include <cmath>
#include <limits>

#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Matrices/matrix_3x3.h"

namespace Physika{

template <typename Scalar>
CPU_GPU_FUNC_DECL SquareMatrix<Scalar,3>::SquareMatrix()
    :SquareMatrix(0) //delegating ctor
{
}

template <typename Scalar>
CPU_GPU_FUNC_DECL SquareMatrix<Scalar,3>::SquareMatrix(Scalar value)
    :SquareMatrix(value, value, value, value, value, value, value, value, value) //delegating ctor
{
}

template <typename Scalar>
CPU_GPU_FUNC_DECL SquareMatrix<Scalar,3>::SquareMatrix(Scalar x00, Scalar x01, Scalar x02, Scalar x10, Scalar x11, Scalar x12, Scalar x20, Scalar x21, Scalar x22)
    :data_(x00, x10, x20,
           x01, x11, x21, 
           x02, x12, x22)
{

}

template <typename Scalar>
CPU_GPU_FUNC_DECL SquareMatrix<Scalar,3>::SquareMatrix(const Vector<Scalar,3> &row1, const Vector<Scalar,3> &row2, const Vector<Scalar,3> &row3)
    :data_(row1[0], row2[0], row3[0], 
           row1[1], row2[1], row3[1], 
           row1[2], row2[2], row3[2])
{

}

template <typename Scalar>
CPU_GPU_FUNC_DECL Scalar& SquareMatrix<Scalar,3>::operator() (unsigned int i, unsigned int j)
{
    return const_cast<Scalar &>(static_cast<const SquareMatrix<Scalar, 3> &>(*this)(i, j));
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Scalar& SquareMatrix<Scalar,3>::operator() (unsigned int i, unsigned int j) const
{
#ifndef __CUDA_ARCH__
    bool index_valid = (i < 3) && (j < 3);
    if (!index_valid)
        throw PhysikaException("Matrix index out of range!");
#endif
    //Note: glm is column-based
    return data_[j][i];
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Vector<Scalar,3> SquareMatrix<Scalar,3>::rowVector(unsigned int i) const
{
#ifndef __CUDA_ARCH__
    if(i>=3)
        throw PhysikaException("Matrix index out of range!");
#endif
    Vector<Scalar,3> result((*this)(i,0),(*this)(i,1),(*this)(i,2));
    return result;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Vector<Scalar,3> SquareMatrix<Scalar,3>::colVector(unsigned int i) const
{
#ifndef __CUDA_ARCH__
    if(i>=3)
        throw PhysikaException("Matrix index out of range!");
#endif
    Vector<Scalar,3> result((*this)(0,i),(*this)(1,i),(*this)(2,i));
    return result;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,3> SquareMatrix<Scalar,3>::operator+ (const SquareMatrix<Scalar,3> &mat2) const
{
    return SquareMatrix<Scalar, 3>(*this) += mat2;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL SquareMatrix<Scalar,3>& SquareMatrix<Scalar,3>::operator+= (const SquareMatrix<Scalar,3> &mat2)
{
    data_ += mat2.data_;
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,3> SquareMatrix<Scalar,3>::operator- (const SquareMatrix<Scalar,3> &mat2) const
{
    return SquareMatrix<Scalar, 3>(*this) -= mat2;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL SquareMatrix<Scalar,3>& SquareMatrix<Scalar,3>::operator-= (const SquareMatrix<Scalar,3> &mat2)
{
    data_ -= mat2.data_;
    return *this;
}


template <typename Scalar>
CPU_GPU_FUNC_DECL bool SquareMatrix<Scalar,3>::operator== (const SquareMatrix<Scalar,3> &mat2) const
{
    return data_ == mat2.data_;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL bool SquareMatrix<Scalar,3>::operator!= (const SquareMatrix<Scalar,3> &mat2) const
{
    return !((*this)==mat2);
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,3> SquareMatrix<Scalar,3>::operator* (Scalar scale) const
{
    return SquareMatrix<Scalar, 3>(*this) *= scale;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL SquareMatrix<Scalar,3>& SquareMatrix<Scalar,3>::operator*= (Scalar scale)
{
    data_ *= scale;
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const Vector<Scalar,3> SquareMatrix<Scalar,3>::operator* (const Vector<Scalar,3> &vec) const
{
    Vector<Scalar,3> result(0);
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            result[i] += (*this)(i, j)*vec[j];
    return result;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,3> SquareMatrix<Scalar,3>::operator* (const SquareMatrix<Scalar,3> &mat2) const
{
    return SquareMatrix<Scalar, 3>(*this) *= mat2;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL SquareMatrix<Scalar,3>& SquareMatrix<Scalar,3>::operator*= (const SquareMatrix<Scalar,3> &mat2)
{
    data_ *= mat2.data_;
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,3> SquareMatrix<Scalar,3>::operator/ (Scalar scale) const
{
    return SquareMatrix<Scalar, 3>(*this) /= scale;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL SquareMatrix<Scalar,3>& SquareMatrix<Scalar,3>::operator/= (Scalar scale)
{
#ifndef __CUDA_ARCH__
    if (abs(scale) <= std::numeric_limits<Scalar>::epsilon())
        throw PhysikaException("Matrix Divide by zero error!");
#endif
    data_ /= scale;
    return *this;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const SquareMatrix<Scalar, 3> SquareMatrix<Scalar, 3>::operator- (void) const
{
    SquareMatrix<Scalar, 3> res;
    res.data_ = -data_;
    return res;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,3> SquareMatrix<Scalar,3>::transpose() const
{
    SquareMatrix<Scalar, 3> res;
    res.data_ = glm::transpose(data_);
    return res;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,3> SquareMatrix<Scalar,3>::inverse() const
{
    SquareMatrix<Scalar, 3> res;
    res.data_ = glm::inverse(data_);

#ifndef __CUDA_ARCH__
    double fir_ele = static_cast<double>(res(0, 0));
    if (std::isnan(fir_ele) == true || std::isinf(fir_ele) == true)
        throw PhysikaException("Matrix not invertible!");
#endif

    return res;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Scalar SquareMatrix<Scalar,3>::determinant() const
{
    return glm::determinant(data_);
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Scalar SquareMatrix<Scalar,3>::trace() const
{
    return (*this)(0,0) + (*this)(1,1) + (*this)(2,2);
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Scalar SquareMatrix<Scalar,3>::doubleContraction(const SquareMatrix<Scalar,3> &mat2) const
{
    Scalar result = 0;
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            result += (*this)(i, j)*mat2(i, j);
    return result;
}

template <typename Scalar>
CPU_GPU_FUNC_DECL Scalar SquareMatrix<Scalar,3>::frobeniusNorm() const
{
    Scalar result = 0;
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
            result += (*this)(i, j)*(*this)(i, j);
    return glm::sqrt(result);
}

template <typename Scalar>
CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,3> SquareMatrix<Scalar,3>::identityMatrix()
{
    return SquareMatrix<Scalar, 3>(1.0, 0.0, 0.0, 
                                   0.0, 1.0, 0.0, 
                                   0.0, 0.0, 1.0);
}

//explicit instantiation of template so that it could be compiled into a lib
template class SquareMatrix<unsigned char, 3>;
template class SquareMatrix<unsigned short, 3>;
template class SquareMatrix<unsigned int, 3>;
template class SquareMatrix<unsigned long, 3>;
template class SquareMatrix<unsigned long long, 3>;
template class SquareMatrix<signed char, 3>;
template class SquareMatrix<short, 3>;
template class SquareMatrix<int, 3>;
template class SquareMatrix<long, 3>;
template class SquareMatrix<long long, 3>;
template class SquareMatrix<float,3>;
template class SquareMatrix<double,3>;
template class SquareMatrix<long double,3>;

}  //end of namespace Physika
