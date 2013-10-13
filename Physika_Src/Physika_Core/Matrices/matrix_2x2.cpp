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

#include <iostream>
#include "Physika_Core/Matrices/matrix_2x2.h"

namespace Physika{

template <typename Scalar>
Matrix2x2<Scalar>::Matrix2x2()
{
}

template <typename Scalar>
Matrix2x2<Scalar>::Matrix2x2(const Scalar x00, const Scalar x01, const Scalar x10, const Scalar x11)
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
  eigen_matrix_2x2_(0,0) = x00;
  eigen_matrix_2x2_(0,1) = x01;
  eigen_matrix_2x2_(1,0) = x10;
  eigen_matrix_2x2_(1,1) = x11;
#endif
}

template <typename Scalar>
Matrix2x2<Scalar>::~Matrix2x2()
{
}

template <typename Scalar>
Scalar& Matrix2x2<Scalar>::operator() (int i, int j)
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
  return eigen_matrix_2x2_(i,j);
#endif
}

template <typename Scalar>
const Scalar& Matrix2x2<Scalar>::operator() (int i, int j) const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
  return eigen_matrix_2x2_(i,j);
#endif
}

template <typename Scalar>
Matrix2x2<Scalar> Matrix2x2<Scalar>::operator+ (const Matrix2x2<Scalar> &mat2) const
{
  Scalar result[4];
  for(int i = 0; i < 2; ++i)
    for(int j = 0; j < 2; ++j)
      result[i+2*j] = (*this)(i,j) + mat2(i,j);
  return Matrix2x2<Scalar>(result[0], result[1], result[2], result[3]);
}

template <typename Scalar>
Matrix2x2<Scalar>& Matrix2x2<Scalar>::operator+= (const Matrix2x2<Scalar> &mat2)
{
  for(int i = 0; i < 2; ++i)
    for(int j = 0; j < 2; ++j)
      (*this)(i,j) = (*this)(i,j) + mat2(i,j);
  return *this;
}

template <typename Scalar>
Matrix2x2<Scalar> Matrix2x2<Scalar>::operator- (const Matrix2x2<Scalar> &mat2) const
{
  Scalar result[4];
  for(int i = 0; i < 2; ++i)
    for(int j = 0; j < 2; ++j)
      result[i+2*j] = (*this)(i,j) - mat2(i,j);
  return Matrix2x2<Scalar>(result[0], result[1], result[2], result[3]);
}

template <typename Scalar>
Matrix2x2<Scalar>& Matrix2x2<Scalar>::operator-= (const Matrix2x2<Scalar> &mat2)
{
  for(int i = 0; i < 2; ++i)
    for(int j = 0; j < 2; ++j)
      (*this)(i,j) = (*this)(i,j) - mat2(i,j);
  return *this;
}

template <typename Scalar>
Matrix2x2<Scalar>& Matrix2x2<Scalar>::operator= (const Matrix2x2<Scalar> &mat2)
{
  for(int i = 0; i < 2; ++i)
    for(int j = 0; j < 2; ++j)
      (*this)(i,j) = mat2(i,j);
  return *this;
}

template <typename Scalar>
bool Matrix2x2<Scalar>::operator== (const Matrix2x2<Scalar> &mat2) const
{
  for(int i = 0; i < 2; ++i)
    for(int j = 0; j < 2; ++j)
      if((*this)(i,j) != mat2(i,j))
	return false;
  return true;
}

template <typename Scalar>
Matrix2x2<Scalar> Matrix2x2<Scalar>::operator* (Scalar scale) const
{
  Scalar result[4];
  for(int i = 0; i < 2; ++i)
    for(int j = 0; j < 2; ++j)
      result[i+2*j] = (*this)(i,j) * scale;
  return Matrix2x2<Scalar>(result[0], result[1], result[2], result[3]);
}

template <typename Scalar>
Matrix2x2<Scalar>& Matrix2x2<Scalar>::operator*= (Scalar scale)
{
  for(int i = 0; i < 2; ++i)
    for(int j = 0; j < 2; ++j)
      (*this)(i,j) = (*this)(i,j) * scale;
  return *this;
}

template <typename Scalar>
Matrix2x2<Scalar> Matrix2x2<Scalar>::operator/ (Scalar scale) const
{
  Scalar result[4];
  for(int i = 0; i < 2; ++i)
    for(int j = 0; j < 2; ++j)
      result[i+2*j] = (*this)(i,j) / scale;
  return Matrix2x2<Scalar>(result[0], result[1], result[2], result[3]);
}

template <typename Scalar>
Matrix2x2<Scalar>& Matrix2x2<Scalar>::operator/= (Scalar scale)
{
  for(int i = 0; i < 2; ++i)
    for(int j = 0; j < 2; ++j)
      (*this)(i,j) = (*this)(i,j) / scale;
  return *this;
}

template <typename Scalar>
Matrix2x2<Scalar> Matrix2x2<Scalar>::transpose() const
{
  Scalar result[4];
  for(int i = 0; i < 2; ++i)
    for(int j = 0; j< 2; ++j)
      result[i+2*j] = (*this)(j,i);
  return Matrix2x2<Scalar>(result[0], result[1], result[2], result[3]);
}

template <typename Scalar>
Matrix2x2<Scalar> Matrix2x2<Scalar>::inverse() const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
  Eigen::Matrix<Scalar,2,2> result_matrix = eigen_matrix_2x2_.inverse();
  return Matrix2x2<Scalar>(result_matrix(0,0), result_matrix(0,1), result_matrix(1,0), result_matrix(1,1));
#endif 
}

template <typename Scalar>
Scalar Matrix2x2<Scalar>::determinant() const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
  return eigen_matrix_2x2_.determinant();
#endif
}

//explicit instantiation of template so that it could be compiled into a lib
template class Matrix2x2<float>;
template class Matrix2x2<double>;

}  //end of namespace Physika
