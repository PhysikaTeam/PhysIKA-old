/*
 * @file matrix_3x3.cpp 
 * @brief 3x3 matrix.
 * @author Sheng Yang, Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <cmath>
#include <limits>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Matrices/matrix_3x3.h"

namespace Physika{

template <typename Scalar>
SquareMatrix<Scalar,3>::SquareMatrix()
{
}

template <typename Scalar>
SquareMatrix<Scalar,3>::SquareMatrix(Scalar x00, Scalar x01, Scalar x02, Scalar x10, Scalar x11, Scalar x12, Scalar x20, Scalar x21, Scalar x22)
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    eigen_matrix_3x3_(0,0) = x00;
    eigen_matrix_3x3_(0,1) = x01;
    eigen_matrix_3x3_(0,2) = x02;
    eigen_matrix_3x3_(1,0) = x10;
    eigen_matrix_3x3_(1,1) = x11;
    eigen_matrix_3x3_(1,2) = x12;
    eigen_matrix_3x3_(2,0) = x20;
    eigen_matrix_3x3_(2,1) = x21;
    eigen_matrix_3x3_(2,2) = x22;
#endif
}

template <typename Scalar>
SquareMatrix<Scalar,3>::SquareMatrix(const Vector<Scalar,3> &row1, const Vector<Scalar,3> &row2, const Vector<Scalar,3> &row3)
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    for(int col = 0; col < 3; ++col)
    {
	eigen_matrix_3x3_(0,col) = row1[col];
	eigen_matrix_3x3_(1,col) = row2[col];
	eigen_matrix_3x3_(2,col) = row3[col];
    }
#endif
}

template <typename Scalar>
SquareMatrix<Scalar,3>::SquareMatrix(const SquareMatrix<Scalar,3> &mat2)
{
    *this = mat2;
}

template <typename Scalar>
SquareMatrix<Scalar,3>::~SquareMatrix()
{
}

template <typename Scalar>
Scalar& SquareMatrix<Scalar,3>::operator() (int i, int j)
{
    PHYSIKA_ASSERT(i>=0&&i<3);
    PHYSIKA_ASSERT(j>=0&&j<3);
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return eigen_matrix_3x3_(i,j);
#endif
}

template <typename Scalar>
const Scalar& SquareMatrix<Scalar,3>::operator() (int i, int j) const
{
    PHYSIKA_ASSERT(i>=0&&i<3);
    PHYSIKA_ASSERT(j>=0&&j<3);
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return eigen_matrix_3x3_(i,j);
#endif
}

template <typename Scalar>
SquareMatrix<Scalar,3> SquareMatrix<Scalar,3>::operator+ (const SquareMatrix<Scalar,3> &mat3) const
{
    Scalar result[9];
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            result[i*3+j] = (*this)(i,j) + mat3(i,j);
    return SquareMatrix<Scalar,3>(result[0], result[1], result[2], result[3] , result[4], result[5], result[6], result[7], result[8]);
}

template <typename Scalar>
SquareMatrix<Scalar,3>& SquareMatrix<Scalar,3>::operator+= (const SquareMatrix<Scalar,3> &mat3)
{
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            (*this)(i,j) = (*this)(i,j) + mat3(i,j);
    return *this;
}

template <typename Scalar>
SquareMatrix<Scalar,3> SquareMatrix<Scalar,3>::operator- (const SquareMatrix<Scalar,3> &mat3) const
{
    Scalar result[9];
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            result[i*3+j] = (*this)(i,j) - mat3(i,j);
    return SquareMatrix<Scalar,3>(result[0], result[1], result[2], result[3] , result[4], result[5], result[6], result[7], result[8]);
}

template <typename Scalar>
SquareMatrix<Scalar,3>& SquareMatrix<Scalar,3>::operator-= (const SquareMatrix<Scalar,3> &mat3)
{
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            (*this)(i,j) = (*this)(i,j) - mat3(i,j);
    return *this;
}

template <typename Scalar>
SquareMatrix<Scalar,3>& SquareMatrix<Scalar,3>::operator= (const SquareMatrix<Scalar,3> &mat3)
{
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            (*this)(i,j) = mat3(i,j);
    return *this;
}

template <typename Scalar>
bool SquareMatrix<Scalar,3>::operator== (const SquareMatrix<Scalar,3> &mat3) const
{
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            if((*this)(i,j) != mat3(i,j))
	        return false;
    return true;
}

template <typename Scalar>
SquareMatrix<Scalar,3> SquareMatrix<Scalar,3>::operator* (Scalar scale) const
{
    Scalar result[9];
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            result[i*3+j] = (*this)(i,j) * scale;
    return SquareMatrix<Scalar,3>(result[0], result[1], result[2], result[3] , result[4], result[5], result[6], result[7], result[8]);
}

template <typename Scalar>
SquareMatrix<Scalar,3>& SquareMatrix<Scalar,3>::operator*= (Scalar scale)
{
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            (*this)(i,j) = (*this)(i,j) * scale;
    return *this;
}

template <typename Scalar>
Vector<Scalar,3> SquareMatrix<Scalar,3>::operator* (const Vector<Scalar,3> &vec) const
{
    Vector<Scalar,3> result(0);
    for(int i = 0; i < 3; ++i)
	for(int j = 0; j < 3; ++j)
	    result[i] += (*this)(i,j)*vec[j];
    return result;
}

template <typename Scalar>
SquareMatrix<Scalar,3> SquareMatrix<Scalar,3>::operator* (const SquareMatrix<Scalar,3> &mat2) const
{
    SquareMatrix<Scalar,3> result(0,0,0,0,0,0,0,0,0);
    for(int i = 0; i < 3; ++i)
	for(int j = 0; j < 3; ++j)
	    for(int k = 0; k < 3; ++k)
		result(i,j) += (*this)(i,k) * mat2(k,j);
    return result;
}

template <typename Scalar>
SquareMatrix<Scalar,3> SquareMatrix<Scalar,3>::operator/ (Scalar scale) const
{
    PHYSIKA_ASSERT(fabs(scale)>std::numeric_limits<Scalar>::epsilon());
    Scalar result[9];
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            result[i*3+j] = (*this)(i,j) / scale;
    return SquareMatrix<Scalar,3>(result[0], result[1], result[2], result[3] , result[4], result[5], result[6], result[7], result[8]);
}

template <typename Scalar>
SquareMatrix<Scalar,3>& SquareMatrix<Scalar,3>::operator/= (Scalar scale)
{
    PHYSIKA_ASSERT(fabs(scale)>std::numeric_limits<Scalar>::epsilon());
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            (*this)(i,j) = (*this)(i,j) / scale;
    return *this;
}

template <typename Scalar>
SquareMatrix<Scalar,3> SquareMatrix<Scalar,3>::transpose() const
{
    Scalar result[9];
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j< 3; ++j)
            result[i*3+j] = (*this)(j,i);
    return SquareMatrix<Scalar,3>(result[0], result[1], result[2], result[3] , result[4], result[5], result[6], result[7], result[8]);
}

template <typename Scalar>
SquareMatrix<Scalar,3> SquareMatrix<Scalar,3>::inverse() const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    Eigen::Matrix<Scalar,3,3> result_matrix = eigen_matrix_3x3_.inverse();
    return SquareMatrix<Scalar,3>(result_matrix(0,0), result_matrix(0,1), result_matrix(0,2), result_matrix(1,0), result_matrix(1,1),result_matrix(1,2), result_matrix(2,0),result_matrix(2,1), result_matrix(2,2));
#endif 
}

template <typename Scalar>
Scalar SquareMatrix<Scalar,3>::determinant() const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return eigen_matrix_3x3_.determinant();
#endif
}

template <typename Scalar>
Scalar SquareMatrix<Scalar,3>::trace() const
{
    return (*this)(0,0) + (*this)(1,1) + (*this)(2,2);
}

template <typename Scalar>
SquareMatrix<Scalar,3> SquareMatrix<Scalar,3>::identityMatrix()
{
    return SquareMatrix<Scalar,3>(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0);
}

template <typename Scalar>
Scalar SquareMatrix<Scalar,3>::doubleContraction(const SquareMatrix<Scalar,3> &mat2) const
{
    Scalar result = 0;
    for(int i = 0; i < 3; ++i)
	for(int j = 0; j < 3; ++j)
	    result += (*this)(i,j)*mat2(i,j);
    return result;
}

//explicit instantiation of template so that it could be compiled into a lib
template class SquareMatrix<float,3>;
template class SquareMatrix<double,3>;

}  //end of namespace Physika
