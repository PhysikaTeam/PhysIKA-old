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

#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Matrices/matrix_3x3.h"

namespace Physika{

template <typename Scalar>
Matrix3x3<Scalar>::Matrix3x3()
{
}

template <typename Scalar>
Matrix3x3<Scalar>::Matrix3x3(Scalar x00, Scalar x01, Scalar x02, Scalar x10, Scalar x11, Scalar x12, Scalar x20, Scalar x21, Scalar x22)
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
Matrix3x3<Scalar>::Matrix3x3(const Matrix3x3<Scalar> &mat2)
{
    *this = mat2;
}

template <typename Scalar>
Matrix3x3<Scalar>::~Matrix3x3()
{
}

template <typename Scalar>
Scalar& Matrix3x3<Scalar>::operator() (int i, int j)
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return eigen_matrix_3x3_(i,j);
#endif
}

template <typename Scalar>
const Scalar& Matrix3x3<Scalar>::operator() (int i, int j) const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return eigen_matrix_3x3_(i,j);
#endif
}

template <typename Scalar>
Matrix3x3<Scalar> Matrix3x3<Scalar>::operator+ (const Matrix3x3<Scalar> &mat3) const
{
    Scalar result[9];
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            result[i*3+j] = (*this)(i,j) + mat3(i,j);
    return Matrix3x3<Scalar>(result[0], result[1], result[2], result[3] , result[4], result[5], result[6], result[7], result[8]);
}

template <typename Scalar>
Matrix3x3<Scalar>& Matrix3x3<Scalar>::operator+= (const Matrix3x3<Scalar> &mat3)
{
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            (*this)(i,j) = (*this)(i,j) + mat3(i,j);
    return *this;
}

template <typename Scalar>
Matrix3x3<Scalar> Matrix3x3<Scalar>::operator- (const Matrix3x3<Scalar> &mat3) const
{
    Scalar result[9];
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            result[i*3+j] = (*this)(i,j) - mat3(i,j);
    return Matrix3x3<Scalar>(result[0], result[1], result[2], result[3] , result[4], result[5], result[6], result[7], result[8]);
}

template <typename Scalar>
Matrix3x3<Scalar>& Matrix3x3<Scalar>::operator-= (const Matrix3x3<Scalar> &mat3)
{
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            (*this)(i,j) = (*this)(i,j) - mat3(i,j);
    return *this;
}

template <typename Scalar>
Matrix3x3<Scalar>& Matrix3x3<Scalar>::operator= (const Matrix3x3<Scalar> &mat3)
{
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            (*this)(i,j) = mat3(i,j);
    return *this;
}

template <typename Scalar>
bool Matrix3x3<Scalar>::operator== (const Matrix3x3<Scalar> &mat3) const
{
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            if((*this)(i,j) != mat3(i,j))
	        return false;
    return true;
}

template <typename Scalar>
Matrix3x3<Scalar> Matrix3x3<Scalar>::operator* (Scalar scale) const
{
    Scalar result[9];
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            result[i*3+j] = (*this)(i,j) * scale;
    return Matrix3x3<Scalar>(result[0], result[1], result[2], result[3] , result[4], result[5], result[6], result[7], result[8]);
}

template <typename Scalar>
Matrix3x3<Scalar>& Matrix3x3<Scalar>::operator*= (Scalar scale)
{
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            (*this)(i,j) = (*this)(i,j) * scale;
    return *this;
}

template <typename Scalar>
Vector3D<Scalar> Matrix3x3<Scalar>::operator* (const Vector3D<Scalar> &vec) const
{
    Vector3D<Scalar> result(0);
    for(int i = 0; i < 3; ++i)
	for(int j = 0; j < 3; ++j)
	    result[i] += (*this)(i,j)*vec[j];
    return result;
}

template <typename Scalar>
Matrix3x3<Scalar> Matrix3x3<Scalar>::operator* (const Matrix3x3<Scalar> &mat2) const
{
    Matrix3x3<Scalar> result(0,0,0,0,0,0,0,0,0);
    for(int i = 0; i < 3; ++i)
	for(int j = 0; j < 3; ++j)
	    for(int k = 0; k < 3; ++k)
		result(i,j) += (*this)(i,k) * mat2(k,j);
    return result;
}

template <typename Scalar>
Matrix3x3<Scalar> Matrix3x3<Scalar>::operator/ (Scalar scale) const
{
    Scalar result[9];
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            result[i*3+j] = (*this)(i,j) / scale;
    return Matrix3x3<Scalar>(result[0], result[1], result[2], result[3] , result[4], result[5], result[6], result[7], result[8]);
}

template <typename Scalar>
Matrix3x3<Scalar>& Matrix3x3<Scalar>::operator/= (Scalar scale)
{
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            (*this)(i,j) = (*this)(i,j) / scale;
    return *this;
}

template <typename Scalar>
Matrix3x3<Scalar> Matrix3x3<Scalar>::transpose() const
{
    Scalar result[9];
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j< 3; ++j)
            result[i*3+j] = (*this)(j,i);
    return Matrix3x3<Scalar>(result[0], result[1], result[2], result[3] , result[4], result[5], result[6], result[7], result[8]);
}

template <typename Scalar>
Matrix3x3<Scalar> Matrix3x3<Scalar>::inverse() const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    Eigen::Matrix<Scalar,3,3> result_matrix = eigen_matrix_3x3_.inverse();
    return Matrix3x3<Scalar>(result_matrix(0,0), result_matrix(0,1), result_matrix(0,2), result_matrix(1,0), result_matrix(1,1),result_matrix(1,2), result_matrix(2,0),result_matrix(2,1), result_matrix(2,2));
#endif 
}

template <typename Scalar>
Scalar Matrix3x3<Scalar>::determinant() const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return eigen_matrix_3x3_.determinant();
#endif
}

template <typename Scalar>
Scalar Matrix3x3<Scalar>::trace() const
{
    return (*this)(0,0) + (*this)(1,1) + (*this)(2,2);
}

//explicit instantiation of template so that it could be compiled into a lib
template class Matrix3x3<float>;
template class Matrix3x3<double>;

}  //end of namespace Physika
