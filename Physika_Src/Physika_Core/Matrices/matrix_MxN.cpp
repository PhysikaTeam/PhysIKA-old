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

#include <cmath>
#include <limits>
#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
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
    PHYSIKA_ASSERT(rows>=0&&cols>=0);
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    ptr_eigen_matrix_MxN_ = new Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>(rows,cols);
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    data_ = new Scalar[rows*cols];
    rows_ = rows;
    cols_ = cols;
 #endif
}

template <typename Scalar>
MatrixMxN<Scalar>::~MatrixMxN()
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    delete ptr_eigen_matrix_MxN_;
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    delete[] data_;
#endif
}

template <typename Scalar>
int MatrixMxN<Scalar>::rows() const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return (*ptr_eigen_matrix_MxN_).rows();
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    return rows_;
#endif
}

template <typename Scalar>
int MatrixMxN<Scalar>::cols() const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return (*ptr_eigen_matrix_MxN_).cols();
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    return cols_;
#endif
}

template <typename Scalar>
void MatrixMxN<Scalar>::resize(int new_rows, int new_cols)
{
    PHYSIKA_ASSERT(new_rows>=0&&new_cols>=0);
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    (*ptr_eigen_matrix_MxN_).resize(new_rows,new_cols);
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    delete[] data_;
    allocMemory(new_rows, new_cols);
#endif
}

template <typename Scalar>
Scalar& MatrixMxN<Scalar>::operator() (int i, int j)
{
    PHYSIKA_ASSERT(i>=0&&i<(*this).rows());
    PHYSIKA_ASSERT(j>=0&&j<(*this).cols());
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return (*ptr_eigen_matrix_MxN_)(i,j);
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    return data_[i*cols_+j];
#endif
}

template <typename Scalar>
const Scalar& MatrixMxN<Scalar>::operator() (int i, int j) const
{
    PHYSIKA_ASSERT(i>=0&&i<(*this).rows());
    PHYSIKA_ASSERT(j>=0&&j<(*this).cols());
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return (*ptr_eigen_matrix_MxN_)(i,j);
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    return data_[i*cols_+j];
#endif
}

template <typename Scalar>
MatrixMxN<Scalar> MatrixMxN<Scalar>::operator+ (const MatrixMxN<Scalar> &mat2) const
{
    int rows = (*this).rows();
    int cols = (*this).cols();
    int rows2 = mat2.rows();
    int cols2 = mat2.cols();
    PHYSIKA_ASSERT((rows==rows2)&&(cols==cols2));
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
    PHYSIKA_ASSERT((rows==rows2)&&(cols==cols2));
    for(int i = 0; i < rows; ++i)
        for(int j = 0; j <cols; ++j)
            (*this)(i,j) = (*this)(i,j) + mat2(i,j);
    return *this;
}

template <typename Scalar>
MatrixMxN<Scalar> MatrixMxN<Scalar>::operator- (const MatrixMxN<Scalar> &mat2) const
{
    int rows = (*this).rows();
    int cols = (*this).cols();
    int rows2 = mat2.rows();
    int cols2 = mat2.cols();
    PHYSIKA_ASSERT((rows==rows2)&&(cols==cols2));
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
    PHYSIKA_ASSERT((rows==rows2)&&(cols==cols2));
    for(int i = 0; i < rows; ++i)
        for(int j = 0; j <cols; ++j)
            (*this)(i,j) = (*this)(i,j) - mat2(i,j);
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
	(*this).resize(rows2,cols2);
    for(int i = 0; i < rows2; ++i)
        for(int j = 0; j < cols2; ++j)
            (*this)(i,j) = mat2(i,j);
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
VectorND<Scalar> MatrixMxN<Scalar>::operator* (const VectorND<Scalar> &vec) const
{
    int mat_row = (*this).rows();
    int mat_col = (*this).cols();
    int vec_dim = vec.dims();
    PHYSIKA_ASSERT(mat_col==vec_dim);
    VectorND<Scalar> result(mat_row,0.0);
    for(int i = 0; i < mat_row; ++i)
    {
	for(int j = 0; j < mat_col; ++j)
	    result[i] += (*this)(i,j)*vec[j];
    }
    return result;
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
    PHYSIKA_ASSERT(fabs(scale)>std::numeric_limits<Scalar>::epsilon());
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
    PHYSIKA_ASSERT(fabs(scale)>std::numeric_limits<Scalar>::epsilon());
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
    PHYSIKA_ASSERT(rows==cols);
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> result_eigen_matrix = (*ptr_eigen_matrix_MxN_).inverse();
    Scalar *result = new Scalar[rows*cols];
    for(int i = 0; i < rows; ++i)
        for(int j = 0; j < cols; ++j)
            result[i*cols+j] = result_eigen_matrix(i,j);
    MatrixMxN<Scalar> result_matrix(rows,cols,result);
    delete result;
    return result_matrix;
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    Scalar det = determinant();
    PHYSIKA_ASSERT(det != 0);
    MatrixMxN<Scalar> result = cofactorMatrix();
    result = result.transpose();
    result /= det;
    return result;
#endif
}

template <typename Scalar>
MatrixMxN<Scalar> MatrixMxN<Scalar>::cofactorMatrix() const
{
    int rows = (*this).rows();
    int cols = (*this).cols();
    PHYSIKA_ASSERT(rows==cols);
    MatrixMxN<Scalar> mat(rows,cols);
    for(int i = 0; i < rows; ++i)
	for(int j = 0; j < cols; ++j)
	{
	    MatrixMxN<Scalar> sub_mat(rows-1,cols-1);
	    for(int ii = 0; ii < rows; ++ii)
		for(int jj =0; jj< cols; ++jj)
		{
		    if((ii==i)||(jj==j)) continue;
		    int row_idx = ii>i?ii-1:ii;
		    int col_idx = jj>j?jj-1:jj;
		    sub_mat(row_idx,col_idx) = (*this)(ii,jj);
		}
	    mat(i,j)=sub_mat.determinant();
	    if((i+j)%2)
		mat(i,j)*=-1;
	}
    return mat;
}

template <typename Scalar>
Scalar MatrixMxN<Scalar>::determinant() const
{
    int rows = (*this).rows();
    int cols = (*this).cols();
    PHYSIKA_ASSERT(rows==cols);
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return (*ptr_eigen_matrix_MxN_).determinant();
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    Scalar det = 0.0;
    for(int j = 0; j < cols; ++j)
    {
	MatrixMxN<Scalar> sub_mat(rows-1,cols-1);
	for(int ii = 1; ii < rows; ++ii)
	    for(int jj =0; jj< cols; ++jj)
	    {
		if((jj==j)) continue;
		int row_idx = ii-1;
		int col_idx = jj>j?jj-1:jj;
		sub_mat(row_idx,col_idx) = (*this)(ii,jj);
	    }
	if(j%2)
	    det -= (*this)(0,j)*sub_mat.determinant();
	else
	    det += (*this)(0,j)*sub_mat.determinant();
    }
    return det;
#endif
}

template <typename Scalar>
Scalar MatrixMxN<Scalar>::trace() const
{
    int rows = (*this).rows();
    int cols = (*this).cols();
    PHYSIKA_ASSERT(rows==cols);
    Scalar result = 0.0;
    for(int i = 0; i < rows; ++i)
	result += (*this)(i,i);
    return result;
}

template <typename Scalar>
Scalar MatrixMxN<Scalar>::doubleContraction(const MatrixMxN<Scalar> &mat2) const
{
    int row1 = (*this).rows();
    int col1 = (*this).cols();
    int row2 = mat2.rows();
    int col2 = mat2.cols();
    PHYSIKA_ASSERT(row1==row2);
    PHYSIKA_ASSERT(col1==col2);
    Scalar result = 0.0;
    for(int i = 0; i < row1; ++i)
	for(int j = 0; j < col1; ++j)
	    result += (*this)(i,j)*mat2(i,j);
    return result;
}

//explicit instantiation of template so that it could be compiled into a lib
template class MatrixMxN<float>;
template class MatrixMxN<double>;

}  //end of namespace Physika
