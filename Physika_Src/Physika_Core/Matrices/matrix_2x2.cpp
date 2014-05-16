/*
 * @file matrix_2x2.cpp 
 * @brief 2x2 matrix.
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
#include <iostream>
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Matrices/matrix_2x2.h"

namespace Physika{

template <typename Scalar>
SquareMatrix<Scalar,2>::SquareMatrix()
{
}

template <typename Scalar>
SquareMatrix<Scalar,2>::SquareMatrix(Scalar x00, Scalar x01, Scalar x10, Scalar x11)
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    eigen_matrix_2x2_(0,0) = x00;
    eigen_matrix_2x2_(0,1) = x01;
    eigen_matrix_2x2_(1,0) = x10;
    eigen_matrix_2x2_(1,1) = x11;
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    data_[0][0]=x00;
    data_[0][1]=x01;
    data_[1][0]=x10;
    data_[1][1]=x11;
#endif
}

template <typename Scalar>
SquareMatrix<Scalar,2>::SquareMatrix(const Vector<Scalar,2> &row1, const Vector<Scalar,2> &row2)
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    for(int col = 0; col < 2; ++col)
    {
        eigen_matrix_2x2_(0,col) = row1[col];
        eigen_matrix_2x2_(1,col) = row2[col];
    }
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    for(int col = 0; col < 2; ++col)
    {
        data_[0][col] = row1[col];
        data_[1][col] = row2[col];
    }
#endif
}

template <typename Scalar>
SquareMatrix<Scalar,2>::SquareMatrix(const SquareMatrix<Scalar,2> &mat2)
{
    *this = mat2;
}

template <typename Scalar>
SquareMatrix<Scalar,2>::~SquareMatrix()
{
}

template <typename Scalar>
Scalar& SquareMatrix<Scalar,2>::operator() (int i, int j)
{
    bool index_valid = (i>=0&&i<2)&&(j>=0&&j<2);
    if(!index_valid)
    {
        std::cerr<<"Matrix index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return eigen_matrix_2x2_(i,j);
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    return data_[i][j];
#endif
}

template <typename Scalar>
const Scalar& SquareMatrix<Scalar,2>::operator() (int i, int j) const
{
    bool index_valid = (i>=0&&i<2)&&(j>=0&&j<2);
    if(!index_valid)
    {
        std::cerr<<"Matrix index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return eigen_matrix_2x2_(i,j);
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    return data_[i][j];
#endif
}

template <typename Scalar>
SquareMatrix<Scalar,2> SquareMatrix<Scalar,2>::operator+ (const SquareMatrix<Scalar,2> &mat2) const
{
    Scalar result[4];
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            result[i*2+j] = (*this)(i,j) + mat2(i,j);
    return SquareMatrix<Scalar,2>(result[0], result[1], result[2], result[3]);
}

template <typename Scalar>
SquareMatrix<Scalar,2>& SquareMatrix<Scalar,2>::operator+= (const SquareMatrix<Scalar,2> &mat2)
{
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            (*this)(i,j) = (*this)(i,j) + mat2(i,j);
    return *this;
}

template <typename Scalar>
SquareMatrix<Scalar,2> SquareMatrix<Scalar,2>::operator- (const SquareMatrix<Scalar,2> &mat2) const
{
    Scalar result[4];
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            result[i*2+j] = (*this)(i,j) - mat2(i,j);
    return SquareMatrix<Scalar,2>(result[0], result[1], result[2], result[3]);
}

template <typename Scalar>
SquareMatrix<Scalar,2>& SquareMatrix<Scalar,2>::operator-= (const SquareMatrix<Scalar,2> &mat2)
{
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            (*this)(i,j) = (*this)(i,j) - mat2(i,j);
    return *this;
}

template <typename Scalar>
SquareMatrix<Scalar,2>& SquareMatrix<Scalar,2>::operator= (const SquareMatrix<Scalar,2> &mat2)
{
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            (*this)(i,j) = mat2(i,j);
    return *this;
}

template <typename Scalar>
bool SquareMatrix<Scalar,2>::operator== (const SquareMatrix<Scalar,2> &mat2) const
{
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            if((*this)(i,j) != mat2(i,j))
                return false;
    return true;
}

template <typename Scalar>
SquareMatrix<Scalar,2> SquareMatrix<Scalar,2>::operator* (Scalar scale) const
{
    Scalar result[4];
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            result[i*2+j] = (*this)(i,j) * scale;
    return SquareMatrix<Scalar,2>(result[0], result[1], result[2], result[3]);
}

template <typename Scalar>
SquareMatrix<Scalar,2>& SquareMatrix<Scalar,2>::operator*= (Scalar scale)
{
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            (*this)(i,j) = (*this)(i,j) * scale;
    return *this;
}

template <typename Scalar>
Vector<Scalar,2> SquareMatrix<Scalar,2>::operator* (const Vector<Scalar,2> &vec) const
{
    Vector<Scalar,2> result(0);
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j <2; ++j)
            result[i] += (*this)(i,j) * vec[j];
    return result;
}

template <typename Scalar>
SquareMatrix<Scalar,2> SquareMatrix<Scalar,2>::operator* (const SquareMatrix<Scalar,2> &mat2) const
{
    SquareMatrix<Scalar,2> result(0,0,0,0);
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            for(int k = 0; k < 2; ++k)
                result(i,j) += (*this)(i,k) * mat2(k,j);
    return result;
}

template <typename Scalar>
SquareMatrix<Scalar,2> SquareMatrix<Scalar,2>::operator/ (Scalar scale) const
{
    if(abs(scale)<std::numeric_limits<Scalar>::epsilon())
    {
        std::cerr<<"Matrix Divide by zero error!\n";
        std::exit(EXIT_FAILURE);
    }
    Scalar result[4];
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            result[i*2+j] = (*this)(i,j) / scale;
    return SquareMatrix<Scalar,2>(result[0], result[1], result[2], result[3]);
}

template <typename Scalar>
SquareMatrix<Scalar,2>& SquareMatrix<Scalar,2>::operator/= (Scalar scale)
{
    if(abs(scale)<std::numeric_limits<Scalar>::epsilon())
    {
        std::cerr<<"Matrix Divide by zero error!\n";
        std::exit(EXIT_FAILURE);
    }
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            (*this)(i,j) = (*this)(i,j) / scale;
    return *this;
}

template <typename Scalar>
SquareMatrix<Scalar,2> SquareMatrix<Scalar,2>::transpose() const
{
    Scalar result[4];
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j< 2; ++j)
            result[i*2+j] = (*this)(j,i);
    return SquareMatrix<Scalar,2>(result[0], result[1], result[2], result[3]);
}

template <typename Scalar>
SquareMatrix<Scalar,2> SquareMatrix<Scalar,2>::inverse() const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    Eigen::Matrix<Scalar,2,2> result_matrix = eigen_matrix_2x2_.inverse();
    return SquareMatrix<Scalar,2>(result_matrix(0,0), result_matrix(0,1), result_matrix(1,0), result_matrix(1,1));
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    Scalar det = determinant();
    if(det==0)
    {
        std::cerr<<"Matrix not invertible!\n";
        std::exit(EXIT_FAILURE);
    }
    return SquareMatrix<Scalar,2>(data_[1][1]/det, -data_[0][1]/det, -data_[1][0]/det, data_[0][0]/det);
#endif 
}

template <typename Scalar>
Scalar SquareMatrix<Scalar,2>::determinant() const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return eigen_matrix_2x2_.determinant();
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    return data_[0][0]*data_[1][1]-data_[0][1]*data_[1][0];
#endif
}

template <typename Scalar>
Scalar SquareMatrix<Scalar,2>::trace() const
{
    return (*this)(0,0) + (*this)(1,1);
}

template <typename Scalar>
SquareMatrix<Scalar,2> SquareMatrix<Scalar,2>::identityMatrix()
{
    return SquareMatrix<Scalar,2>(1.0,0.0,0.0,1.0);
}

template <typename Scalar>
Scalar SquareMatrix<Scalar,2>::doubleContraction(const SquareMatrix<Scalar,2> &mat2) const
{
    Scalar result = 0;
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            result += (*this)(i,j)*mat2(i,j);
    return result;
}

//explicit instantiation of template so that it could be compiled into a lib
template class SquareMatrix<float,2>;
template class SquareMatrix<double,2>;
template class SquareMatrix<int,2>;

}  //end of namespace Physika
