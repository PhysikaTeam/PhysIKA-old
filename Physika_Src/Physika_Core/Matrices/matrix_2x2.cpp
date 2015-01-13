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
#include <cstdlib>
#include <iostream>
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Matrices/matrix_2x2.h"

namespace Physika{

template <typename Scalar>
SquareMatrix<Scalar,2>::SquareMatrix()
{
}

template <typename Scalar>
SquareMatrix<Scalar,2>::SquareMatrix(Scalar value)
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    eigen_matrix_2x2_(0,0) = value;
    eigen_matrix_2x2_(0,1) = value;
    eigen_matrix_2x2_(1,0) = value;
    eigen_matrix_2x2_(1,1) = value;
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    data_[0][0]=value;
    data_[0][1]=value;
    data_[1][0]=value;
    data_[1][1]=value;
#endif
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
    for(unsigned int col = 0; col < 2; ++col)
    {
        eigen_matrix_2x2_(0,col) = row1[col];
        eigen_matrix_2x2_(1,col) = row2[col];
    }
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    for(unsigned int col = 0; col < 2; ++col)
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
Scalar& SquareMatrix<Scalar,2>::operator() (unsigned int i, unsigned int j)
{
    bool index_valid = (i<2)&&(j<2);
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
const Scalar& SquareMatrix<Scalar,2>::operator() (unsigned int i, unsigned int j) const
{
    bool index_valid = (i<2)&&(j<2);
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
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            result[i*2+j] = (*this)(i,j) + mat2(i,j);
    return SquareMatrix<Scalar,2>(result[0], result[1], result[2], result[3]);
}

template <typename Scalar>
SquareMatrix<Scalar,2>& SquareMatrix<Scalar,2>::operator+= (const SquareMatrix<Scalar,2> &mat2)
{
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            (*this)(i,j) = (*this)(i,j) + mat2(i,j);
    return *this;
}

template <typename Scalar>
SquareMatrix<Scalar,2> SquareMatrix<Scalar,2>::operator- (const SquareMatrix<Scalar,2> &mat2) const
{
    Scalar result[4];
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            result[i*2+j] = (*this)(i,j) - mat2(i,j);
    return SquareMatrix<Scalar,2>(result[0], result[1], result[2], result[3]);
}

template <typename Scalar>
SquareMatrix<Scalar,2>& SquareMatrix<Scalar,2>::operator-= (const SquareMatrix<Scalar,2> &mat2)
{
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            (*this)(i,j) = (*this)(i,j) - mat2(i,j);
    return *this;
}

template <typename Scalar>
SquareMatrix<Scalar,2>& SquareMatrix<Scalar,2>::operator= (const SquareMatrix<Scalar,2> &mat2)
{
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            (*this)(i,j) = mat2(i,j);
    return *this;
}

template <typename Scalar>
bool SquareMatrix<Scalar,2>::operator== (const SquareMatrix<Scalar,2> &mat2) const
{
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
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
bool SquareMatrix<Scalar,2>::operator!= (const SquareMatrix<Scalar,2> &mat2) const
{
    return !((*this)==mat2);
}

template <typename Scalar>
SquareMatrix<Scalar,2> SquareMatrix<Scalar,2>::operator* (Scalar scale) const
{
    Scalar result[4];
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            result[i*2+j] = (*this)(i,j) * scale;
    return SquareMatrix<Scalar,2>(result[0], result[1], result[2], result[3]);
}

template <typename Scalar>
SquareMatrix<Scalar,2>& SquareMatrix<Scalar,2>::operator*= (Scalar scale)
{
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            (*this)(i,j) = (*this)(i,j) * scale;
    return *this;
}

template <typename Scalar>
Vector<Scalar,2> SquareMatrix<Scalar,2>::operator* (const Vector<Scalar,2> &vec) const
{
    Vector<Scalar,2> result(0);
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j <2; ++j)
            result[i] += (*this)(i,j) * vec[j];
    return result;
}

template <typename Scalar>
SquareMatrix<Scalar,2> SquareMatrix<Scalar,2>::operator* (const SquareMatrix<Scalar,2> &mat2) const
{
    SquareMatrix<Scalar,2> result(0,0,0,0);
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            for(unsigned int k = 0; k < 2; ++k)
                result(i,j) += (*this)(i,k) * mat2(k,j);
    return result;
}

template <typename Scalar>
SquareMatrix<Scalar,2>& SquareMatrix<Scalar,2>::operator*= (const SquareMatrix<Scalar,2> &mat2)
{
    SquareMatrix<Scalar,2> result(0,0,0,0);
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            for(unsigned int k = 0; k < 2; ++k)
                result(i,j) += (*this)(i,k) * mat2(k,j);
    *this = result;
    return *this;
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
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
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
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            (*this)(i,j) = (*this)(i,j) / scale;
    return *this;
}

template <typename Scalar>
SquareMatrix<Scalar,2> SquareMatrix<Scalar,2>::transpose() const
{
    Scalar result[4];
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j< 2; ++j)
            result[i*2+j] = (*this)(j,i);
    return SquareMatrix<Scalar,2>(result[0], result[1], result[2], result[3]);
}

template <typename Scalar>
SquareMatrix<Scalar,2> SquareMatrix<Scalar,2>::inverse() const
{
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
    {
        std::cerr<<"Matrix not invertible!\n";
        std::exit(EXIT_FAILURE);
    }
    return SquareMatrix<Scalar,2>((*this)(1,1)/det, -(*this)(0,1)/det, -(*this)(1,0)/det, (*this)(0,0)/det);
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
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            result += (*this)(i,j)*mat2(i,j);
    return result;
}

template <typename Scalar>
void SquareMatrix<Scalar,2>::singularValueDecomposition(SquareMatrix<Scalar,2> &left_singular_vectors,
                                                        Vector<Scalar,2> &singular_values,
                                                        SquareMatrix<Scalar,2> &right_singular_vectors) const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    //hack: Eigen::SVD does not support integer types, hence we cast Scalar to long double for decomposition
    Eigen::Matrix<long double,2,2> temp_matrix;
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)          
            temp_matrix(i,j) = static_cast<long double>(eigen_matrix_2x2_(i,j));
    Eigen::JacobiSVD<Eigen::Matrix<long double,2,2> > svd(temp_matrix,Eigen::ComputeThinU|Eigen::ComputeThinV);
    const Eigen::Matrix<long double,2,2> &left = svd.matrixU(), &right = svd.matrixV();
    const Eigen::Matrix<long double,2,1> &values = svd.singularValues();
    for(unsigned int i = 0; i < 2; ++i)
    {
        singular_values[i] = static_cast<Scalar>(values(i,0));
        for(unsigned int j = 0; j < 2; ++j)
        {
            left_singular_vectors(i,j) = static_cast<Scalar>(left(i,j));
            right_singular_vectors(i,j) = static_cast<Scalar>(right(i,j));
        }
    }
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    std::cerr<<"SVD not implemeted for built in matrix!\n";
    std::exit(EXIT_FAILURE);
#endif
}

template <typename Scalar>
void SquareMatrix<Scalar,2>::eigenDecomposition(Vector<Scalar,2> &eigen_values_real, Vector<Scalar,2> &eigen_values_imag,
                                                SquareMatrix<Scalar,2> &eigen_vectors_real, SquareMatrix<Scalar,2> &eigen_vectors_imag)
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    //hack: Eigen::EigenSolver does not support integer types, hence we cast Scalar to long double for decomposition
    Eigen::Matrix<long double,2,2> temp_matrix;
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)          
            temp_matrix(i,j) = static_cast<long double>(eigen_matrix_2x2_(i,j));
    Eigen::EigenSolver<Eigen::Matrix<long double,2,2> > eigen(temp_matrix);
    Eigen::Matrix<std::complex<long double>,2,2> vectors = eigen.eigenvectors();
    const Eigen::Matrix<std::complex<long double>,2,1> &values = eigen.eigenvalues();
    for(unsigned int i = 0; i < 2; ++i)
    {
        eigen_values_real[i] = static_cast<Scalar>(values(i,0).real());
        eigen_values_imag[i] = static_cast<Scalar>(values(i,0).imag());
        for(unsigned int j = 0; j < 2; ++j)
        {
            eigen_vectors_real(i,j) = static_cast<Scalar>(vectors(i,j).real());
            eigen_vectors_imag(i,j) = static_cast<Scalar>(vectors(i,j).imag());
        }
    }
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    std::cerr<<"Eigen decomposition not implemeted for built in matrix!\n";
    std::exit(EXIT_FAILURE);
#endif
}

//explicit instantiation of template so that it could be compiled into a lib
template class SquareMatrix<unsigned char,2>;
template class SquareMatrix<unsigned short,2>;
template class SquareMatrix<unsigned int,2>;
template class SquareMatrix<unsigned long,2>;
template class SquareMatrix<unsigned long long,2>;
template class SquareMatrix<signed char,2>;
template class SquareMatrix<short,2>;
template class SquareMatrix<int,2>;
template class SquareMatrix<long,2>;
template class SquareMatrix<long long,2>;
template class SquareMatrix<float,2>;
template class SquareMatrix<double,2>;
template class SquareMatrix<long double,2>;

}  //end of namespace Physika
