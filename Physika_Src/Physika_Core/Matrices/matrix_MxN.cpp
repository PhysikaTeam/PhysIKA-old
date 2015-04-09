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

#include <limits>
#include <cstdlib>
#include <iostream>
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Vectors/vector_Nd.h"
#include "Physika_Core/Matrices/matrix_MxN.h"

namespace Physika{

template <typename Scalar>
MatrixMxN<Scalar>::MatrixMxN()
{
    allocMemory(0,0);
}

template <typename Scalar>
MatrixMxN<Scalar>::MatrixMxN(unsigned int rows, unsigned int cols)
{
    allocMemory(rows,cols);
}

template <typename Scalar>
MatrixMxN<Scalar>::MatrixMxN(unsigned int rows, unsigned int cols, Scalar value)
{
    allocMemory(rows,cols);
    for(unsigned int i = 0; i < rows; ++i)
        for(unsigned int j = 0; j < cols; ++j)
            (*this)(i,j) = value;
}

template <typename Scalar>
MatrixMxN<Scalar>::MatrixMxN(unsigned int rows, unsigned int cols, Scalar *entries)
{
    allocMemory(rows,cols);
    for(unsigned int i = 0; i < rows; ++i)
        for(unsigned int j = 0; j < cols; ++j)
            (*this)(i,j) = entries[i*cols+j];
}

template <typename Scalar>
MatrixMxN<Scalar>::MatrixMxN(const MatrixMxN<Scalar> &mat2)
{
    allocMemory(mat2.rows(),mat2.cols());
    *this = mat2;
}

template<typename Scalar>
void MatrixMxN<Scalar>::allocMemory(unsigned int rows, unsigned int cols)
{ 
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    ptr_eigen_matrix_MxN_ = new Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>(rows,cols);
    PHYSIKA_ASSERT(ptr_eigen_matrix_MxN_);
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    data_ = new Scalar[rows*cols];
    PHYSIKA_ASSERT(data_);
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
unsigned int MatrixMxN<Scalar>::rows() const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return (*ptr_eigen_matrix_MxN_).rows();
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    return rows_;
#endif
}

template <typename Scalar>
unsigned int MatrixMxN<Scalar>::cols() const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return (*ptr_eigen_matrix_MxN_).cols();
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    return cols_;
#endif
}

template <typename Scalar>
void MatrixMxN<Scalar>::resize(unsigned int new_rows, unsigned int new_cols)
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    (*ptr_eigen_matrix_MxN_).resize(new_rows,new_cols);
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    delete[] data_;
    allocMemory(new_rows, new_cols);
#endif
}

template <typename Scalar>
Scalar& MatrixMxN<Scalar>::operator() (unsigned int i, unsigned int j)
{
    bool index_in_range = (i<(*this).rows())&&(j<(*this).cols());
    if(!index_in_range)
        throw PhysikaException("Matrix index out of range!");
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return (*ptr_eigen_matrix_MxN_)(i,j);
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    return data_[i*cols_+j];
#endif
}

template <typename Scalar>
const Scalar& MatrixMxN<Scalar>::operator() (unsigned int i, unsigned int j) const
{
    bool index_in_range = (i<(*this).rows())&&(j<(*this).cols());
    if(!index_in_range)
        throw PhysikaException("Matrix index out of range!");
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return (*ptr_eigen_matrix_MxN_)(i,j);
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    return data_[i*cols_+j];
#endif
}

template <typename Scalar>
VectorND<Scalar> MatrixMxN<Scalar>::rowVector(unsigned int i) const
{
    if(i>=(*this).rows())
        throw PhysikaException("Matrix index out of range!");
    VectorND<Scalar> result((*this).cols());
    for(unsigned int col = 0; col < result.dims(); ++col)
        result[col] = (*this)(i,col);
    return result;
}

template <typename Scalar>
VectorND<Scalar> MatrixMxN<Scalar>::colVector(unsigned int i) const
{
    if(i>=(*this).cols())
        throw PhysikaException("Matrix index out of range!");
    VectorND<Scalar> result((*this).rows());
    for(unsigned int row = 0; row < result.dims(); ++row)
        result[row] = (*this)(row,i);
    return result;
}

template <typename Scalar>
MatrixMxN<Scalar> MatrixMxN<Scalar>::operator+ (const MatrixMxN<Scalar> &mat2) const
{
    unsigned int rows = (*this).rows();
    unsigned int cols = (*this).cols();
    unsigned int rows2 = mat2.rows();
    unsigned int cols2 = mat2.cols();
    bool size_match = (rows==rows2)&&(cols==cols2);
    if(!size_match)
        throw PhysikaException("Cannot add two matrix of different sizes!");
    Scalar *result = new Scalar[rows*cols];
    for(unsigned int i = 0; i < rows; ++i)
        for(unsigned int j = 0; j < cols; ++j)
            result[i*cols+j] = (*this)(i,j) + mat2(i,j);
    MatrixMxN<Scalar> result_matrix(rows, cols, result);
    delete result;
    return result_matrix;
}

template <typename Scalar>
MatrixMxN<Scalar>& MatrixMxN<Scalar>::operator+= (const MatrixMxN<Scalar> &mat2)
{
    unsigned int rows = (*this).rows();
    unsigned int cols = (*this).cols();
    unsigned int rows2 = mat2.rows();
    unsigned int cols2 = mat2.cols();
    bool size_match = (rows==rows2)&&(cols==cols2);
    if(!size_match)
        throw PhysikaException("Cannot add two matrix of different sizes!");
    for(unsigned int i = 0; i < rows; ++i)
        for(unsigned int j = 0; j <cols; ++j)
            (*this)(i,j) = (*this)(i,j) + mat2(i,j);
    return *this;
}

template <typename Scalar>
MatrixMxN<Scalar> MatrixMxN<Scalar>::operator- (const MatrixMxN<Scalar> &mat2) const
{
    unsigned int rows = (*this).rows();
    unsigned int cols = (*this).cols();
    unsigned int rows2 = mat2.rows();
    unsigned int cols2 = mat2.cols();
    bool size_match = (rows==rows2)&&(cols==cols2);
    if(!size_match)
        throw PhysikaException("Cannot subtract two matrix of different sizes!");
    Scalar *result = new Scalar[rows*cols];
    for(unsigned int i = 0; i < rows; ++i)
        for(unsigned int j = 0; j < cols; ++j)
            result[i*cols+j] = (*this)(i,j) - mat2(i,j);
    MatrixMxN<Scalar> result_matrix(rows, cols, result);
    delete result;
    return result_matrix;
}

template <typename Scalar>
MatrixMxN<Scalar>& MatrixMxN<Scalar>::operator-= (const MatrixMxN<Scalar> &mat2)
{
    unsigned int rows = (*this).rows();
    unsigned int cols = (*this).cols();
    unsigned int rows2 = mat2.rows();
    unsigned int cols2 = mat2.cols();
    bool size_match = (rows==rows2)&&(cols==cols2);
    if(!size_match)
        throw PhysikaException("Cannot subtract two matrix of different sizes!");
    for(unsigned int i = 0; i < rows; ++i)
        for(unsigned int j = 0; j <cols; ++j)
            (*this)(i,j) = (*this)(i,j) - mat2(i,j);
    return *this;
}

template <typename Scalar>
MatrixMxN<Scalar>& MatrixMxN<Scalar>::operator= (const MatrixMxN<Scalar> &mat2)
{
    unsigned int rows = (*this).rows();
    unsigned int cols = (*this).cols();
    unsigned int rows2 = mat2.rows();
    unsigned int cols2 = mat2.cols();
    if((rows != rows2)||(cols != cols2))
        (*this).resize(rows2,cols2);
    for(unsigned int i = 0; i < rows2; ++i)
        for(unsigned int j = 0; j < cols2; ++j)
            (*this)(i,j) = mat2(i,j);
    return *this;
}

template <typename Scalar>
bool MatrixMxN<Scalar>::operator== (const MatrixMxN<Scalar> &mat2) const
{
    unsigned int rows1 = (*this).rows();
    unsigned int cols1 = (*this).cols();
    unsigned int rows2 = mat2.rows();
    unsigned int cols2 = mat2.cols();
    if((rows1 != rows2)||(cols1 != cols2))
        return false;
    for(unsigned int i = 0; i < rows1; ++i)
        for(unsigned int j = 0; j < cols1; ++j)
        {
            if(is_floating_point<Scalar>::value)
            {
                if(isEqual((*this)(i,j),mat2(i,j))==false)
                    return false;
            }
            else
            {
                if((*this)(i,j) != mat2(i,j))
                    return false;
            }
        }
    return true;
}

template <typename Scalar>
bool MatrixMxN<Scalar>::operator!= (const MatrixMxN<Scalar> &mat2) const
{
    return !((*this)==mat2);
}

template <typename Scalar>
MatrixMxN<Scalar> MatrixMxN<Scalar>::operator* (Scalar scale) const
{
    unsigned int rows = (*this).rows();
    unsigned int cols = (*this).cols();
    Scalar *result = new Scalar[rows*cols];
    for(unsigned int i = 0; i < rows; ++i)
        for(unsigned int j = 0; j < cols; ++j)
            result[i*cols+j] = (*this)(i,j) * scale;
    MatrixMxN<Scalar> result_matrix(rows,cols,result);
    delete result;
    return result_matrix;
}

template <typename Scalar>
MatrixMxN<Scalar>& MatrixMxN<Scalar>::operator*= (Scalar scale)
{
    unsigned int rows = (*this).rows();
    unsigned int cols = (*this).cols();
    for(unsigned int i = 0; i < rows; ++i)
        for(unsigned int j = 0; j < cols; ++j)
            (*this)(i,j) = (*this)(i,j) * scale;
    return *this;
}
    
template <typename Scalar>
VectorND<Scalar> MatrixMxN<Scalar>::operator* (const VectorND<Scalar> &vec) const
{
    unsigned int mat_row = (*this).rows();
    unsigned int mat_col = (*this).cols();
    unsigned int vec_dim = vec.dims();
    if(mat_col!=vec_dim)
        throw PhysikaException("Matrix*Vector: Matrix and vector sizes do not match!");
    VectorND<Scalar> result(mat_row,0.0);
    for(unsigned int i = 0; i < mat_row; ++i)
    {
        for(unsigned int j = 0; j < mat_col; ++j)
            result[i] += (*this)(i,j)*vec[j];
    }
    return result;
}

template <typename Scalar>
MatrixMxN<Scalar> MatrixMxN<Scalar>::operator* (const MatrixMxN<Scalar> &mat2) const
{
    unsigned int mat1_row = (*this).rows();
    unsigned int mat1_col = (*this).cols();
    unsigned int mat2_row = mat2.rows();
    unsigned int mat2_col = mat2.cols();
    if(mat1_col != mat2_row)
        throw PhysikaException("Matrix*Matrix: Matrix and matrix sizes do not match!");
    MatrixMxN<Scalar> result(mat1_row,mat2_col,static_cast<Scalar>(0));
    for(unsigned int i = 0; i < mat1_row; ++i)
        for(unsigned int j = 0; j < mat2_col; ++j)
            for(unsigned int k = 0; k < mat1_col; ++k)
                result(i,j) += (*this)(i,k)*mat2(k,j);
    return result;
}

template <typename Scalar>
MatrixMxN<Scalar>& MatrixMxN<Scalar>::operator*= (const MatrixMxN<Scalar> &mat2)
{
    unsigned int mat1_row = (*this).rows();
    unsigned int mat1_col = (*this).cols();
    unsigned int mat2_row = mat2.rows();
    unsigned int mat2_col = mat2.cols();
    if(mat1_col != mat2_row)
        throw PhysikaException("Matrix*Matrix: Matrix and matrix sizes do not match!");
    MatrixMxN<Scalar> result(mat1_row,mat2_col,static_cast<Scalar>(0));
    for(unsigned int i = 0; i < mat1_row; ++i)
        for(unsigned int j = 0; j < mat2_col; ++j)
            for(unsigned int k = 0; k < mat1_col; ++k)
                result(i,j) += (*this)(i,k)*mat2(k,j);
    *this = result;
    return *this;
}
    
template <typename Scalar>
MatrixMxN<Scalar> MatrixMxN<Scalar>::operator/ (Scalar scale) const
{
    if(abs(scale)<std::numeric_limits<Scalar>::epsilon())
        throw PhysikaException("Matrix Divide by zero error!");
    unsigned int rows = (*this).rows();
    unsigned int cols = (*this).cols();
    Scalar *result = new Scalar[rows*cols];
    for(unsigned int i = 0; i < rows; ++i)
        for(unsigned int j = 0; j < cols; ++j)
            result[i*cols+j] = (*this)(i,j) / scale;
    MatrixMxN<Scalar> result_matrix(rows,cols,result);
    delete result;
    return result_matrix;
}

template <typename Scalar>
MatrixMxN<Scalar>& MatrixMxN<Scalar>::operator/= (Scalar scale)
{
    if(abs(scale)<std::numeric_limits<Scalar>::epsilon())
        throw PhysikaException("Matrix Divide by zero error!");
    unsigned int rows = (*this).rows();
    unsigned int cols = (*this).cols();
    for(unsigned int i = 0; i < rows; ++i)
        for(unsigned int j = 0; j < cols; ++j)
            (*this)(i,j) = (*this)(i,j) / scale;
    return *this;
}

template <typename Scalar>
MatrixMxN<Scalar> MatrixMxN<Scalar>::transpose() const
{
    unsigned int rows = (*this).rows();
    unsigned int cols = (*this).cols();
    Scalar *result = new Scalar[rows*cols];
    for(unsigned int i = 0; i < rows; ++i)
        for(unsigned int j = 0; j < cols; ++j)
            result[j*rows+i] = (*this)(i,j);
    MatrixMxN<Scalar> result_matrix(cols,rows,result);
    delete result;
    return result_matrix;
}

template <typename Scalar>
MatrixMxN<Scalar> MatrixMxN<Scalar>::inverse() const
{
    unsigned int rows = (*this).rows();
    unsigned int cols = (*this).cols();
    if(rows!=cols)
        throw PhysikaException("Matrix not square matrix, it's not invertible!");
    Scalar det = determinant();
    bool singular = false;
    if(is_floating_point<Scalar>::value)
    {
        if(isEqual(det,static_cast<Scalar>(0)))
            singular = true;
    }
    else
    {
        if(det == 0)
            singular = true;
    }
    if(singular)
        throw PhysikaException("Matrix not invertible!");
    MatrixMxN<Scalar> result = cofactorMatrix();
    result = result.transpose();
    result /= det;
    return result;
}

template <typename Scalar>
MatrixMxN<Scalar> MatrixMxN<Scalar>::cofactorMatrix() const
{
    unsigned int rows = (*this).rows();
    unsigned int cols = (*this).cols();
    if(rows!=cols)
        throw PhysikaException("Matrix not square matrix, cofactor matrix does not exit!");
    MatrixMxN<Scalar> mat(rows,cols);
    for(unsigned int i = 0; i < rows; ++i)
        for(unsigned int j = 0; j < cols; ++j)
        {
            MatrixMxN<Scalar> sub_mat(rows-1,cols-1);
            for(unsigned int ii = 0; ii < rows; ++ii)
                for(unsigned int jj =0; jj< cols; ++jj)
                {
                    if((ii==i)||(jj==j)) continue;
                    unsigned int row_idx = ii>i?ii-1:ii;
                    unsigned int col_idx = jj>j?jj-1:jj;
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
    unsigned int rows = (*this).rows();
    unsigned int cols = (*this).cols();
    if(rows!=cols)
        throw PhysikaException("Matrix not square matrix, determinant does not exit!");
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return (*ptr_eigen_matrix_MxN_).determinant();
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    Scalar det = 0.0;
    if(rows==1)
        return (*this)(0,0);
    for(unsigned int j = 0; j < cols; ++j)
    {
        MatrixMxN<Scalar> sub_mat(rows-1,cols-1);
        for(unsigned int ii = 1; ii < rows; ++ii)
            for(unsigned int jj =0; jj< cols; ++jj)
            {
                if(jj==j) continue;
                unsigned int row_idx = ii-1;
                unsigned int col_idx = jj>j?jj-1:jj;
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
    unsigned int rows = (*this).rows();
    unsigned int cols = (*this).cols();
    if(rows!=cols)
        throw PhysikaException("Matrix not square matrix, trace does not exit!");
    Scalar result = 0.0;
    for(unsigned int i = 0; i < rows; ++i)
        result += (*this)(i,i);
    return result;
}

template <typename Scalar>
Scalar MatrixMxN<Scalar>::doubleContraction(const MatrixMxN<Scalar> &mat2) const
{
    unsigned int row1 = (*this).rows();
    unsigned int col1 = (*this).cols();
    unsigned int row2 = mat2.rows();
    unsigned int col2 = mat2.cols();
    bool size_match = (row1==row2)&&(col1==col2);
    if(!size_match)
        throw PhysikaException("Cannot compute double contraction of two matrix with different sizes!");
    Scalar result = 0.0;
    for(unsigned int i = 0; i < row1; ++i)
        for(unsigned int j = 0; j < col1; ++j)
            result += (*this)(i,j)*mat2(i,j);
    return result;
}

template <typename Scalar>
void MatrixMxN<Scalar>::singularValueDecomposition(MatrixMxN<Scalar> &left_singular_vectors,
                                                   VectorND<Scalar> &singular_values,
                                                   MatrixMxN<Scalar> &right_singular_vectors) const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    //hack: Eigen::SVD does not support integer types, hence we cast Scalar to long double for decomposition
    unsigned int rows = this->rows(), cols = this->cols();
    Eigen::Matrix<long double,Eigen::Dynamic,Eigen::Dynamic> temp_matrix(rows,cols);
    for(unsigned int i = 0; i < rows; ++i)
        for(unsigned int j = 0; j < cols; ++j)          
                temp_matrix(i,j) = static_cast<long double>((*ptr_eigen_matrix_MxN_)(i,j));
    Eigen::JacobiSVD<Eigen::Matrix<long double,Eigen::Dynamic,Eigen::Dynamic> > svd(temp_matrix,Eigen::ComputeThinU|Eigen::ComputeThinV);
    const Eigen::Matrix<long double,Eigen::Dynamic,Eigen::Dynamic> &left = svd.matrixU(), &right = svd.matrixV();
    const Eigen::Matrix<long double,Eigen::Dynamic,1> &values = svd.singularValues();
    //resize if have to
    if(left_singular_vectors.rows() != left.rows() || left_singular_vectors.cols() != left.cols())
        left_singular_vectors.resize(left.rows(),left.cols());
    if(right_singular_vectors.rows() != right.rows() || right_singular_vectors.cols() != right.cols())
        right_singular_vectors.resize(right.rows(),right.cols());
    if(singular_values.dims() != values.rows())
        singular_values.resize(values.rows());
    //copy the result
    for(unsigned int i = 0; i < left.rows(); ++i)
        for(unsigned int j = 0; j < left.cols(); ++j)
            left_singular_vectors(i,j) = static_cast<Scalar>(left(i,j));
    for(unsigned int i = 0; i < right.rows(); ++i)
        for(unsigned int j = 0; j < right.cols(); ++j)
            right_singular_vectors(i,j) = static_cast<Scalar>(right(i,j));
    for(unsigned int i = 0; i < values.rows(); ++i)
        singular_values[i] = static_cast<Scalar>(values(i,0));
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    throw PhysikaException("SVD not implemeted for built in matrix!");
#endif
}

template <typename Scalar>
void MatrixMxN<Scalar>::eigenDecomposition(VectorND<Scalar> &eigen_values_real, VectorND<Scalar> &eigen_values_imag,
                                           MatrixMxN<Scalar> &eigen_vectors_real, MatrixMxN<Scalar> &eigen_vectors_imag)
{
    unsigned int rows = this->rows(), cols = this->cols();
    if(rows != cols)
        throw PhysikaException("Eigen decomposition is only valid for square matrix!");
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    //hack: Eigen::EigenSolver does not support integer types, hence we cast Scalar to long double for decomposition
    Eigen::Matrix<long double,Eigen::Dynamic,Eigen::Dynamic> temp_matrix(rows,cols);
    for(unsigned int i = 0; i < rows; ++i)
        for(unsigned int j = 0; j < cols; ++j)          
                temp_matrix(i,j) = static_cast<long double>((*ptr_eigen_matrix_MxN_)(i,j));
    Eigen::EigenSolver<Eigen::Matrix<long double,Eigen::Dynamic,Eigen::Dynamic> > eigen(temp_matrix);
    Eigen::Matrix<std::complex<long double>,Eigen::Dynamic,Eigen::Dynamic> vectors = eigen.eigenvectors();
    const Eigen::Matrix<std::complex<long double>,Eigen::Dynamic,1> &values = eigen.eigenvalues();
    //resize if have to
    if(eigen_vectors_real.rows() != vectors.rows() || eigen_vectors_real.cols() != vectors.cols())
        eigen_vectors_real.resize(vectors.rows(),vectors.cols());
    if(eigen_vectors_imag.rows() != vectors.rows() || eigen_vectors_imag.cols() != vectors.cols())
        eigen_vectors_imag.resize(vectors.rows(),vectors.cols());
    if(eigen_values_real.dims() != values.rows())
        eigen_values_real.resize(values.rows());
    if(eigen_values_imag.dims() != values.rows())
        eigen_values_imag.resize(values.rows());
    //copy the result
    for(unsigned int i = 0; i < vectors.rows(); ++i)
        for(unsigned int j = 0; j < vectors.cols(); ++j)
        {
            eigen_vectors_real(i,j) = static_cast<Scalar>(vectors(i,j).real());
            eigen_vectors_imag(i,j) = static_cast<Scalar>(vectors(i,j).imag());
        }
    for(unsigned int i = 0; i < values.rows(); ++i)
    {
        eigen_values_real[i] = static_cast<Scalar>(values(i,0).real());
        eigen_values_imag[i] = static_cast<Scalar>(values(i,0).imag());
    }
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    throw PhysikaException("Eigen decomposition not implemeted for built in matrix!");
#endif
}

//explicit instantiation of template so that it could be compiled into a lib
template class MatrixMxN<unsigned char>;
template class MatrixMxN<unsigned short>;
template class MatrixMxN<unsigned int>;
template class MatrixMxN<unsigned long>;
template class MatrixMxN<unsigned long long>;
template class MatrixMxN<signed char>;
template class MatrixMxN<short>;
template class MatrixMxN<int>;
template class MatrixMxN<long>;
template class MatrixMxN<long long>;
template class MatrixMxN<float>;
template class MatrixMxN<double>;
template class MatrixMxN<long double>;

}  //end of namespace Physika
