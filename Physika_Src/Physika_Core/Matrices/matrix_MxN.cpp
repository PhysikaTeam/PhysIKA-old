/*
 * @file matrix_MxN.cpp 
 * @brief matrix of arbitrary size, and size could be changed during runtime.
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
#include "Physika_Core/Matrices/matrix_MxN.h"

namespace Physika{

template <typename Scalar>
MatrixMxN<Scalar>::MatrixMxN()
{
  allocMemory(0,0);
}

template <typename Scalar>
MatrixMxN<Scalar>::MatrixMxN(int rows, int cols)
{
  allocMemory(rows,cols);
}

template <typename Scalar>
MatrixMxN<Scalar>::MatrixMxN(int rows, int cols, Scalar *entries)
{
  allocMemory(rows,cols);
  for(int i = 0; i < rows; ++i)
     for(int j = 0; j < cols; ++j)
       (*this)(i,j) = entries[i*cols+j];
}

template <typename Scalar>
MatrixMxN<Scalar>::MatrixMxN(const MatrixMxN<Scalar> &mat2)
{
  allocMemory(mat2.rows(),mat2.cols());
  *this = mat2;
}

template<typename Scalar>
void MatrixMxN<Scalar>::allocMemory(int rows, int cols)
{ 
#ifdef PHYSIKA_USE_EIGEN_MATRIX
   ptr_eigen_matrix_MxN_ = new Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>(rows,cols);
 #endif
}

template <typename Scalar>
MatrixMxN<Scalar>::~MatrixMxN()
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
   delete ptr_eigen_matrix_MxN_;
#endif
}

template <typename Scalar>
int MatrixMxN<Scalar>::rows() const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
  return (*ptr_eigen_matrix_MxN_).rows();
#endif
}

template <typename Scalar>
int MatrixMxN<Scalar>::cols() const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
  return (*ptr_eigen_matrix_MxN_).cols();
#endif
}

template <typename Scalar>
void MatrixMxN<Scalar>::resize(int new_rows, int new_cols)
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
  (*ptr_eigen_matrix_MxN_).resize(new_rows,new_cols);
#endif
}

template <typename Scalar>
Scalar& MatrixMxN<Scalar>::operator() (int i, int j)
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
  return (*ptr_eigen_matrix_MxN_)(i,j);
#endif
}

template <typename Scalar>
const Scalar& MatrixMxN<Scalar>::operator() (int i, int j) const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
  return (*ptr_eigen_matrix_MxN_)(i,j);
#endif
}

template <typename Scalar>
MatrixMxN<Scalar> MatrixMxN<Scalar>::operator+ (const MatrixMxN<Scalar> &mat2) const
{
  int rows = (*this).rows();
  int cols = (*this).cols();
  int rows2 = mat2.rows();
  int cols2 = mat2.cols();
  if((rows != rows2)||(cols != cols2))
  {
    std::cout<<"Error: matrix size doesn't match! Returned an empty matrix."<<std::endl;
    return MatrixMxN<Scalar>();
  }
  Scalar *result = new Scalar[rows*cols];
  for(int i = 0; i < rows; ++i)
    for(int j = 0; j < cols; ++j)
      result[i*cols+j] = (*this)(i,j) + mat2(i,j);
  MatrixMxN<Scalar> result_matrix(rows, cols, result);
  delete result;
  return result_matrix;
}

template <typename Scalar>
MatrixMxN<Scalar>& MatrixMxN<Scalar>::operator+= (const MatrixMxN<Scalar> &mat2)
{
  int rows = (*this).rows();
  int cols = (*this).cols();
  int rows2 = mat2.rows();
  int cols2 = mat2.cols();
  if((rows != rows2)||(cols != cols2))
  {
    std::cout<<"Error: matrix size doesn't match! No operation conducted."<<std::endl;
  }
  else
  {
    for(int i = 0; i < rows; ++i)
      for(int j = 0; j <cols; ++j)
        (*this)(i,j) = (*this)(i,j) + mat2(i,j);
  }
  return *this;
}

template <typename Scalar>
MatrixMxN<Scalar> MatrixMxN<Scalar>::operator- (const MatrixMxN<Scalar> &mat2) const
{
  int rows = (*this).rows();
  int cols = (*this).cols();
  int rows2 = mat2.rows();
  int cols2 = mat2.cols();
  if((rows != rows2)||(cols != cols2))
  {
    std::cout<<"Error: matrix size doesn't match! Returned an empty matrix."<<std::endl;
    return MatrixMxN<Scalar>();
  }
  Scalar *result = new Scalar[rows*cols];
  for(int i = 0; i < rows; ++i)
    for(int j = 0; j < cols; ++j)
      result[i*cols+j] = (*this)(i,j) - mat2(i,j);
  MatrixMxN<Scalar> result_matrix(rows, cols, result);
  delete result;
  return result_matrix;
}

template <typename Scalar>
MatrixMxN<Scalar>& MatrixMxN<Scalar>::operator-= (const MatrixMxN<Scalar> &mat2)
{
  int rows = (*this).rows();
  int cols = (*this).cols();
  int rows2 = mat2.rows();
  int cols2 = mat2.cols();
  if((rows != rows2)||(cols != cols2))
  {
    std::cout<<"Error: matrix size doesn't match! No operation conducted."<<std::endl;
  }
  else
  {
    for(int i = 0; i < rows; ++i)
      for(int j = 0; j <cols; ++j)
        (*this)(i,j) = (*this)(i,j) - mat2(i,j);
  }
  return *this;
}

template <typename Scalar>
MatrixMxN<Scalar>& MatrixMxN<Scalar>::operator= (const MatrixMxN<Scalar> &mat2)
{
  int rows = (*this).rows();
  int cols = (*this).cols();
  int rows2 = mat2.rows();
  int cols2 = mat2.cols();
  if((rows != rows2)||(cols != cols2))
  {
    std::cout<<"Error: matrix size doesn't match! No operation conducted."<<std::endl;
  }
  else
  {
    for(int i = 0; i < rows; ++i)
      for(int j = 0; j < cols; ++j)
        (*this)(i,j) = mat2(i,j);
  }
  return *this;
}

template <typename Scalar>
bool MatrixMxN<Scalar>::operator== (const MatrixMxN<Scalar> &mat2) const
{
  int rows1 = (*this).rows();
  int cols1 = (*this).cols();
  int rows2 = mat2.rows();
  int cols2 = mat2.cols();
  if((rows1 != rows2)||(cols1 != cols2))
    return false;
  for(int i = 0; i < rows1; ++i)
    for(int j = 0; j < cols1; ++j)
      if((*this)(i,j) != mat2(i,j))
	return false;
  return true;
}

template <typename Scalar>
MatrixMxN<Scalar> MatrixMxN<Scalar>::operator* (Scalar scale) const
{
  int rows = (*this).rows();
  int cols = (*this).cols();
  Scalar *result = new Scalar[rows*cols];
  for(int i = 0; i < rows; ++i)
    for(int j = 0; j < cols; ++j)
      result[i*cols+j] = (*this)(i,j) * scale;
  MatrixMxN<Scalar> result_matrix(rows,cols,result);
  delete result;
  return result_matrix;
}

template <typename Scalar>
MatrixMxN<Scalar>& MatrixMxN<Scalar>::operator*= (Scalar scale)
{
  int rows = (*this).rows();
  int cols = (*this).cols();
  for(int i = 0; i < rows; ++i)
    for(int j = 0; j < cols; ++j)
      (*this)(i,j) = (*this)(i,j) * scale;
  return *this;
}

template <typename Scalar>
MatrixMxN<Scalar> MatrixMxN<Scalar>::operator/ (Scalar scale) const
{
int rows = (*this).rows();
  int cols = (*this).cols();
  Scalar *result = new Scalar[rows*cols];
  for(int i = 0; i < rows; ++i)
    for(int j = 0; j < cols; ++j)
      result[i*cols+j] = (*this)(i,j) / scale;
  MatrixMxN<Scalar> result_matrix(rows,cols,result);
  delete result;
  return result_matrix;
}

template <typename Scalar>
MatrixMxN<Scalar>& MatrixMxN<Scalar>::operator/= (Scalar scale)
{
  int rows = (*this).rows();
  int cols = (*this).cols();
  for(int i = 0; i < rows; ++i)
    for(int j = 0; j < cols; ++j)
      (*this)(i,j) = (*this)(i,j) / scale;
  return *this;
}

template <typename Scalar>
MatrixMxN<Scalar> MatrixMxN<Scalar>::transpose() const
{
  int rows = (*this).rows();
  int cols = (*this).cols();
  Scalar *result = new Scalar[rows*cols];
  for(int i = 0; i < rows; ++i)
    for(int j = 0; j < cols; ++j)
      result[j*rows+i] = (*this)(i,j);
  MatrixMxN<Scalar> result_matrix(cols,rows,result);
  delete result;
  return result_matrix;
}

template <typename Scalar>
MatrixMxN<Scalar> MatrixMxN<Scalar>::inverse() const
{
  int rows = (*this).rows();
  int cols = (*this).cols();
  if(rows != cols)
  {
    std::cout<<"Error: matrix is not square! Returned original matrix."<<std::endl;
    return *this;
  }
#ifdef PHYSIKA_USE_EIGEN_MATRIX
  Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> result_eigen_matrix = (*ptr_eigen_matrix_MxN_).inverse();
  Scalar *result = new Scalar[rows*cols];
  for(int i = 0; i < rows; ++i)
    for(int j = 0; j < cols; ++j)
      result[i*cols+j] = result_eigen_matrix(i,j);
  MatrixMxN<Scalar> result_matrix(rows,cols,result);
  delete result;
  return result_matrix;
#endif
}

template <typename Scalar>
Scalar MatrixMxN<Scalar>::determinant() const
{
  int rows = (*this).rows();
  int cols = (*this).cols();
  if(rows != cols)
  {
    std::cout<<"Error: matrix is not square! Returned 0."<<std::endl;
    return 0;
  }
#ifdef PHYSIKA_USE_EIGEN_MATRIX
  return (*ptr_eigen_matrix_MxN_).determinant();
#endif
}

//explicit instantiation of template so that it could be compiled into a lib
template class MatrixMxN<float>;
template class MatrixMxN<double>;

}  //end of namespace Physika
