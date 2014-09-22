/*
 * @file matrix_1x1.cpp 
 * @brief 1x1 matrix.
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
#include "Physika_Core/Vectors/vector_1d.h"
#include "Physika_Core/Matrices/matrix_1x1.h"

namespace Physika{

template <typename Scalar>
SquareMatrix<Scalar,1>::SquareMatrix()
{
}

template <typename Scalar>
SquareMatrix<Scalar,1>::SquareMatrix(Scalar value)
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    eigen_matrix_1x1_(0,0) = value;
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    data_ = value;
#endif
}

template <typename Scalar>
SquareMatrix<Scalar,1>::SquareMatrix(const SquareMatrix<Scalar,1> &mat2)
{
    *this = mat2;
}

template <typename Scalar>
SquareMatrix<Scalar,1>::~SquareMatrix()
{
}

template <typename Scalar>
Scalar& SquareMatrix<Scalar,1>::operator() (unsigned int i, unsigned int j)
{
    bool index_valid = (i<1)&&(j<1);
    if(!index_valid)
    {
        std::cerr<<"Matrix index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return eigen_matrix_1x1_(i,j);
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    return data_;
#endif
}

template <typename Scalar>
const Scalar& SquareMatrix<Scalar,1>::operator() (unsigned int i, unsigned int j) const
{
    bool index_valid = (i<1)&&(j<1);
    if(!index_valid)
    {
        std::cerr<<"Matrix index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return eigen_matrix_1x1_(i,j);
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    return data_;
#endif
}

template <typename Scalar>
SquareMatrix<Scalar,1> SquareMatrix<Scalar,1>::operator+ (const SquareMatrix<Scalar,1> &mat2) const
{
    Scalar result;
    result = (*this)(0,0) + mat2(0,0);
    return SquareMatrix<Scalar,1>(result);
}

template <typename Scalar>
SquareMatrix<Scalar,1>& SquareMatrix<Scalar,1>::operator+= (const SquareMatrix<Scalar,1> &mat2)
{
    (*this)(0,0) = (*this)(0,0) + mat2(0,0);
    return *this;
}

template <typename Scalar>
SquareMatrix<Scalar,1> SquareMatrix<Scalar,1>::operator- (const SquareMatrix<Scalar,1> &mat2) const
{
    Scalar result;
    result = (*this)(0,0) - mat2(0,0);
    return SquareMatrix<Scalar,1>(result);
}

template <typename Scalar>
SquareMatrix<Scalar,1>& SquareMatrix<Scalar,1>::operator-= (const SquareMatrix<Scalar,1> &mat2)
{
    (*this)(0,0) = (*this)(0,0) - mat2(0,0);
    return *this;
}

template <typename Scalar>
SquareMatrix<Scalar,1>& SquareMatrix<Scalar,1>::operator= (const SquareMatrix<Scalar,1> &mat2)
{
    (*this)(0,0) = mat2(0,0);
    return *this;
}

template <typename Scalar>
bool SquareMatrix<Scalar,1>::operator== (const SquareMatrix<Scalar,1> &mat2) const
{
    if((*this)(0,0) != mat2(0,0))
        return false;
    return true;
}

template <typename Scalar>
bool SquareMatrix<Scalar,1>::operator!= (const SquareMatrix<Scalar,1> &mat2) const
{
    return !((*this)==mat2);
}

template <typename Scalar>
SquareMatrix<Scalar,1> SquareMatrix<Scalar,1>::operator* (Scalar scale) const
{
    Scalar result;
    result = (*this)(0,0) * scale;
    return SquareMatrix<Scalar,1>(result);
}

template <typename Scalar>
SquareMatrix<Scalar,1>& SquareMatrix<Scalar,1>::operator*= (Scalar scale)
{
    (*this)(0,0) = (*this)(0,0) * scale;
    return *this;
}

template <typename Scalar>
Vector<Scalar,1> SquareMatrix<Scalar,1>::operator* (const Vector<Scalar,1> &vec) const
{
    Vector<Scalar,1> result(0);
    result[0] += (*this)(0,0) * vec[0];
    return result;
}

template <typename Scalar>
SquareMatrix<Scalar,1> SquareMatrix<Scalar,1>::operator* (const SquareMatrix<Scalar,1> &mat2) const
{
    SquareMatrix<Scalar,1> result(0);
    result(0,0) += (*this)(0,0) * mat2(0,0);
    return result;
}

template <typename Scalar>
SquareMatrix<Scalar,1> SquareMatrix<Scalar,1>::operator/ (Scalar scale) const
{
    if(abs(scale)<std::numeric_limits<Scalar>::epsilon())
    {
        std::cerr<<"Matrix Divide by zero error!\n";
        std::exit(EXIT_FAILURE);
    }
    Scalar result;
    result = (*this)(0,0) / scale;
    return SquareMatrix<Scalar,1>(result);
}

template <typename Scalar>
SquareMatrix<Scalar,1>& SquareMatrix<Scalar,1>::operator/= (Scalar scale)
{
    if(abs(scale)<std::numeric_limits<Scalar>::epsilon())
    {
        std::cerr<<"Matrix Divide by zero error!\n";
        std::exit(EXIT_FAILURE);
    }
    (*this)(0,0) = (*this)(0,0) / scale;
    return *this;
}

template <typename Scalar>
SquareMatrix<Scalar,1> SquareMatrix<Scalar,1>::transpose() const
{
    Scalar result;
    result = (*this)(0,0);
    return SquareMatrix<Scalar,1>(result);
}

template <typename Scalar>
SquareMatrix<Scalar,1> SquareMatrix<Scalar,1>::inverse() const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    Eigen::Matrix<Scalar,1,1> result_matrix = eigen_matrix_1x1_.inverse();
    return SquareMatrix<Scalar,1>(result_matrix(0,0));
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    if(data_==0)
    {
        std::cerr<<"Matrix not invertible!\n";
        std::exit(EXIT_FAILURE);
    }
    return SquareMatrix<Scalar,1>(1/data_);
#endif 
}

template <typename Scalar>
Scalar SquareMatrix<Scalar,1>::determinant() const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return eigen_matrix_1x1_.determinant();
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    return data_;
#endif
}

template <typename Scalar>
Scalar SquareMatrix<Scalar,1>::trace() const
{
    return (*this)(0,0);
}

template <typename Scalar>
SquareMatrix<Scalar,1> SquareMatrix<Scalar,1>::identityMatrix()
{
    return SquareMatrix<Scalar,1>(1);
}

template <typename Scalar>
Scalar SquareMatrix<Scalar,1>::doubleContraction(const SquareMatrix<Scalar,1> &mat2) const
{
    Scalar result = 0;
    result += (*this)(0,0)*mat2(0,0);
    return result;
}

//explicit instantiation of template so that it could be compiled into a lib
template class SquareMatrix<unsigned char,1>;
template class SquareMatrix<unsigned short,1>;
template class SquareMatrix<unsigned int,1>;
template class SquareMatrix<unsigned long,1>;
template class SquareMatrix<unsigned long long,1>;
template class SquareMatrix<signed char,1>;
template class SquareMatrix<short,1>;
template class SquareMatrix<int,1>;
template class SquareMatrix<long,1>;
template class SquareMatrix<long long,1>;
template class SquareMatrix<float,1>;
template class SquareMatrix<double,1>;
template class SquareMatrix<long double,1>;

}  //end of namespace Physika
