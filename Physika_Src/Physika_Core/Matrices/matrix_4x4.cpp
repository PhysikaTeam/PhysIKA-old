/*
 * @file matrix_4x4.cpp
 * @brief 4x4 matrix.
 * @author Sheng Yang, Fei Zhu, Liyou Xu
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
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Vectors/vector_4d.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Matrices/matrix_4x4.h"

namespace Physika{

template <typename Scalar>
SquareMatrix<Scalar,4>::SquareMatrix()
{
}

template <typename Scalar>
SquareMatrix<Scalar,4>::SquareMatrix(Scalar value)
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    eigen_matrix_4x4_(0,0) = value;
    eigen_matrix_4x4_(0,1) = value;
    eigen_matrix_4x4_(0,2) = value;
    eigen_matrix_4x4_(0,3) = value;
    eigen_matrix_4x4_(1,0) = value;
    eigen_matrix_4x4_(1,1) = value;
    eigen_matrix_4x4_(1,2) = value;
    eigen_matrix_4x4_(1,3) = value;
    eigen_matrix_4x4_(2,0) = value;
    eigen_matrix_4x4_(2,1) = value;
    eigen_matrix_4x4_(2,2) = value;
    eigen_matrix_4x4_(2,3) = value;
    eigen_matrix_4x4_(3,0) = value;
    eigen_matrix_4x4_(3,1) = value;
    eigen_matrix_4x4_(3,2) = value;
    eigen_matrix_4x4_(3,3) = value;
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    data_[0][0] = value;
    data_[0][1] = value;
    data_[0][2] = value;
    data_[0][3] = value;
    data_[1][0] = value;
    data_[1][1] = value;
    data_[1][2] = value;
    data_[1][3] = value;
    data_[2][0] = value;
    data_[2][1] = value;
    data_[2][2] = value;
    data_[2][3] = value;
    data_[3][0] = value;
    data_[3][1] = value;
    data_[3][2] = value;
    data_[3][3] = value;
#endif
}

template <typename Scalar>
SquareMatrix<Scalar,4>::SquareMatrix(Scalar x00, Scalar x01, Scalar x02, Scalar x03, Scalar x10, Scalar x11, Scalar x12, Scalar x13, Scalar x20, Scalar x21, Scalar x22, Scalar x23, Scalar x30, Scalar x31, Scalar x32, Scalar x33)
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    eigen_matrix_4x4_(0,0) = x00;
    eigen_matrix_4x4_(0,1) = x01;
    eigen_matrix_4x4_(0,2) = x02;
    eigen_matrix_4x4_(0,3) = x03;
    eigen_matrix_4x4_(1,0) = x10;
    eigen_matrix_4x4_(1,1) = x11;
    eigen_matrix_4x4_(1,2) = x12;
    eigen_matrix_4x4_(1,3) = x13;
    eigen_matrix_4x4_(2,0) = x20;
    eigen_matrix_4x4_(2,1) = x21;
    eigen_matrix_4x4_(2,2) = x22;
    eigen_matrix_4x4_(2,3) = x23;
    eigen_matrix_4x4_(3,0) = x30;
    eigen_matrix_4x4_(3,1) = x31;
    eigen_matrix_4x4_(3,2) = x32;
    eigen_matrix_4x4_(3,3) = x33;
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    data_[0][0] = x00;
    data_[0][1] = x01;
    data_[0][2] = x02;
    data_[0][3] = x03;
    data_[1][0] = x10;
    data_[1][1] = x11;
    data_[1][2] = x12;
    data_[1][3] = x13;
    data_[2][0] = x20;
    data_[2][1] = x21;
    data_[2][2] = x22;
    data_[2][3] = x23;
    data_[3][0] = x30;
    data_[3][1] = x31;
    data_[3][2] = x32;
    data_[3][3] = x33;
#endif
}

template <typename Scalar>
SquareMatrix<Scalar,4>::SquareMatrix(const Vector<Scalar,4> &row1, const Vector<Scalar,4> &row2, const Vector<Scalar,4> &row3, const Vector<Scalar, 4> &row4)
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    for(unsigned int col = 0; col < 4; ++col)
    {
        eigen_matrix_4x4_(0,col) = row1[col];
        eigen_matrix_4x4_(1,col) = row2[col];
        eigen_matrix_4x4_(2,col) = row3[col];
        eigen_matrix_4x4_(3,col) = row4[col];
    }
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    for(unsigned int col = 0; col < 4; ++col)
    {
        data_[0][col] = row1[col];
        data_[1][col] = row2[col];
        data_[2][col] = row3[col];
        data_[3][col] = row4[col];
    }
#endif
}

template <typename Scalar>
SquareMatrix<Scalar,4>::SquareMatrix(const SquareMatrix<Scalar,4> &mat2)
{
    *this = mat2;
}

template <typename Scalar>
SquareMatrix<Scalar,4>::~SquareMatrix()
{
}

template <typename Scalar>
Scalar& SquareMatrix<Scalar,4>::operator() (unsigned int i, unsigned int j)
{
    bool index_valid = (i<4)&&(j<4);
    if(!index_valid)
        throw PhysikaException("Matrix index out of range!");
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return eigen_matrix_4x4_(i,j);
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    return data_[i][j];
#endif
}

template <typename Scalar>
const Scalar& SquareMatrix<Scalar,4>::operator() (unsigned int i, unsigned int j) const
{
    bool index_valid = (i<4)&&(j<4);
    if(!index_valid)
        throw PhysikaException("Matrix index out of range!");
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return eigen_matrix_4x4_(i,j);
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    return data_[i][j];
#endif
}

template <typename Scalar>
Vector<Scalar,4> SquareMatrix<Scalar,4>::rowVector(unsigned int i) const
{
    if(i>=4)
        throw PhysikaException("Matrix index out of range!");
    Vector<Scalar,4> result((*this)(i,0),(*this)(i,1),(*this)(i,2),(*this)(i,3));
    return result;
}

template <typename Scalar>
Vector<Scalar,4> SquareMatrix<Scalar,4>::colVector(unsigned int i) const
{
    if(i>=4)
        throw PhysikaException("Matrix index out of range!");
    Vector<Scalar,4> result((*this)(0,i),(*this)(1,i),(*this)(2,i),(*this)(3,i));
    return result;
}

template <typename Scalar>
SquareMatrix<Scalar,4> SquareMatrix<Scalar,4>::operator+ (const SquareMatrix<Scalar,4> &mat2) const
{
    Scalar result[16];
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j < 4; ++j)
            result[i*4+j] = (*this)(i,j) + mat2(i,j);
    return SquareMatrix<Scalar,4>(result[0], result[1], result[2], result[3] , result[4], result[5], result[6], result[7], result[8], result[9], result[10], result[11], result[12] , result[13], result[14], result[15]);
}

template <typename Scalar>
SquareMatrix<Scalar,4>& SquareMatrix<Scalar,4>::operator+= (const SquareMatrix<Scalar,4> &mat2)
{
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j < 4; ++j)
            (*this)(i,j) = (*this)(i,j) + mat2(i,j);
    return *this;
}

template <typename Scalar>
SquareMatrix<Scalar,4> SquareMatrix<Scalar,4>::operator- (const SquareMatrix<Scalar,4> &mat2) const
{
    Scalar result[16];
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j < 4; ++j)
            result[i*4+j] = (*this)(i,j) - mat2(i,j);
    return SquareMatrix<Scalar,4>(result[0], result[1], result[2], result[3] , result[4], result[5], result[6], result[7], result[8], result[9], result[10], result[11], result[12] , result[13], result[14], result[15]);
}

template <typename Scalar>
SquareMatrix<Scalar,4>& SquareMatrix<Scalar,4>::operator-= (const SquareMatrix<Scalar,4> &mat2)
{
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j < 4; ++j)
            (*this)(i,j) = (*this)(i,j) - mat2(i,j);
    return *this;
}

template <typename Scalar>
SquareMatrix<Scalar,4>& SquareMatrix<Scalar,4>::operator= (const SquareMatrix<Scalar,4> &mat2)
{
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j < 4; ++j)
            (*this)(i,j) = mat2(i,j);
    return *this;
}

template <typename Scalar>
bool SquareMatrix<Scalar,4>::operator== (const SquareMatrix<Scalar,4> &mat2) const
{
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j < 4; ++j)
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
bool SquareMatrix<Scalar,4>::operator!= (const SquareMatrix<Scalar,4> &mat2) const
{
    return !((*this)==mat2);
}

template <typename Scalar>
SquareMatrix<Scalar,4> SquareMatrix<Scalar,4>::operator* (Scalar scale) const
{
    Scalar result[16];
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j < 4; ++j)
            result[i*4+j] = (*this)(i,j) * scale;
    return SquareMatrix<Scalar,4>(result[0], result[1], result[2], result[3] , result[4], result[5], result[6], result[7], result[8], result[9], result[10], result[11], result[12] , result[13], result[14], result[15]);
}

template <typename Scalar>
SquareMatrix<Scalar,4>& SquareMatrix<Scalar,4>::operator*= (Scalar scale)
{
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j < 4; ++j)
            (*this)(i,j) = (*this)(i,j) * scale;
    return *this;
}

template <typename Scalar>
Vector<Scalar,4> SquareMatrix<Scalar,4>::operator* (const Vector<Scalar,4> &vec) const
{
    Vector<Scalar,4> result(0);
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j < 4; ++j)
            result[i] += (*this)(i,j)*vec[j];
    return result;
}

template <typename Scalar>
SquareMatrix<Scalar,4> SquareMatrix<Scalar,4>::operator* (const SquareMatrix<Scalar,4> &mat2) const
{
    SquareMatrix<Scalar,4> result(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0);
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j < 4; ++j)
            for(unsigned int k = 0; k < 4; ++k)
                result(i,j) += (*this)(i,k) * mat2(k,j);
    return result;
}

template <typename Scalar>
SquareMatrix<Scalar,4>& SquareMatrix<Scalar,4>::operator*= (const SquareMatrix<Scalar,4> &mat2)
{
    SquareMatrix<Scalar,4> result(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0);
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j < 4; ++j)
            for(unsigned int k = 0; k < 4; ++k)
                result(i,j) += (*this)(i,k) * mat2(k,j);
    *this = result;
    return *this;
}

template <typename Scalar>
SquareMatrix<Scalar,4> SquareMatrix<Scalar,4>::operator/ (Scalar scale) const
{
    if(abs(scale)<std::numeric_limits<Scalar>::epsilon())
        throw PhysikaException("Matrix Divide by zero error!");
    Scalar result[16];
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j < 4; ++j)
            result[i*4+j] = (*this)(i,j) / scale;
    return SquareMatrix<Scalar,4>(result[0], result[1], result[2], result[3] , result[4], result[5], result[6], result[7], result[8], result[9], result[10], result[11], result[12] , result[13], result[14], result[15]);
}

template <typename Scalar>
SquareMatrix<Scalar,4>& SquareMatrix<Scalar,4>::operator/= (Scalar scale)
{
    if(abs(scale)<std::numeric_limits<Scalar>::epsilon())
        throw PhysikaException("Matrix Divide by zero error!");
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j < 4; ++j)
            (*this)(i,j) = (*this)(i,j) / scale;
    return *this;
}

template <typename Scalar>
SquareMatrix<Scalar,4> SquareMatrix<Scalar,4>::transpose() const
{
    Scalar result[16];
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j< 4; ++j)
            result[i*4+j] = (*this)(j,i);
    return SquareMatrix<Scalar,4>(result[0], result[1], result[2], result[3] , result[4], result[5], result[6], result[7], result[8], result[9], result[10], result[11], result[12] , result[13], result[14], result[15]);
}

template <typename Scalar>
SquareMatrix<Scalar,4> SquareMatrix<Scalar,4>::inverse() const
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
        throw PhysikaException("Matrix not invertible!");
    //companion maxtrix
    Scalar x00 = (SquareMatrix<Scalar, 3>((*this)(1,1), (*this)(1,2), (*this)(1,3), (*this)(2,1), (*this)(2,2), (*this)(2,3), (*this)(3,1), (*this)(3,2), (*this)(3,3))).determinant();
    Scalar x01 = - (SquareMatrix<Scalar, 3>((*this)(1,0), (*this)(1,2), (*this)(1,3), (*this)(2,0), (*this)(2,2), (*this)(2,3), (*this)(3,0), (*this)(3,2), (*this)(3,3))).determinant();
    Scalar x02 = (SquareMatrix<Scalar, 3>((*this)(1,0), (*this)(1,1), (*this)(1,3), (*this)(2,0), (*this)(2,1), (*this)(2,3), (*this)(3,0), (*this)(3,1), (*this)(3,3))).determinant();
    Scalar x03 = - (SquareMatrix<Scalar, 3>((*this)(1,0), (*this)(1,1), (*this)(1,2), (*this)(2,0), (*this)(2,1), (*this)(2,2), (*this)(3,0), (*this)(3,1), (*this)(3,2))).determinant();
    Scalar x10 = - (SquareMatrix<Scalar, 3>((*this)(0,1), (*this)(0,2), (*this)(0,3), (*this)(2,1), (*this)(2,2), (*this)(2,3), (*this)(3,1), (*this)(3,2), (*this)(3,3))).determinant();
    Scalar x11 = (SquareMatrix<Scalar, 3>((*this)(0,0), (*this)(0,2), (*this)(0,3), (*this)(2,0), (*this)(2,2), (*this)(2,3), (*this)(3,0), (*this)(3,2), (*this)(3,3))).determinant();
    Scalar x12 = - (SquareMatrix<Scalar, 3>((*this)(0,0), (*this)(0,1), (*this)(0,3), (*this)(2,0), (*this)(2,1), (*this)(2,3), (*this)(3,0), (*this)(3,1), (*this)(3,3))).determinant();
    Scalar x13 = (SquareMatrix<Scalar, 3>((*this)(0,0), (*this)(0,1), (*this)(0,2), (*this)(2,0), (*this)(2,1), (*this)(2,2), (*this)(3,0), (*this)(3,1), (*this)(3,2))).determinant();
    Scalar x20 = (SquareMatrix<Scalar, 3>((*this)(0,1), (*this)(0,2), (*this)(0,3), (*this)(1,1), (*this)(1,2), (*this)(1,3), (*this)(3,1), (*this)(3,2), (*this)(3,3))).determinant();
    Scalar x21 = - (SquareMatrix<Scalar, 3>((*this)(0,0), (*this)(0,2), (*this)(0,3), (*this)(1,0), (*this)(1,2), (*this)(1,3), (*this)(3,0), (*this)(3,2), (*this)(3,3))).determinant();
    Scalar x22 = (SquareMatrix<Scalar, 3>((*this)(0,0), (*this)(0,1), (*this)(0,3), (*this)(1,0), (*this)(1,1), (*this)(1,3), (*this)(3,0), (*this)(3,1), (*this)(3,3))).determinant();
    Scalar x23 = - (SquareMatrix<Scalar, 3>((*this)(0,0), (*this)(0,1), (*this)(0,2), (*this)(1,0), (*this)(1,1), (*this)(1,2), (*this)(3,0), (*this)(3,1), (*this)(3,2))).determinant();
    Scalar x30 = - (SquareMatrix<Scalar, 3>((*this)(0,1), (*this)(0,2), (*this)(0,3), (*this)(1,1), (*this)(1,2), (*this)(1,3), (*this)(2,1), (*this)(2,2), (*this)(2,3))).determinant();
    Scalar x31 = (SquareMatrix<Scalar, 3>((*this)(0,0), (*this)(0,2), (*this)(0,3), (*this)(1,0), (*this)(1,2), (*this)(1,3), (*this)(2,0), (*this)(2,2), (*this)(2,3))).determinant();
    Scalar x32 = - (SquareMatrix<Scalar, 3>((*this)(0,0), (*this)(0,1), (*this)(0,3), (*this)(1,0), (*this)(1,1), (*this)(1,3), (*this)(2,0), (*this)(2,1), (*this)(2,3))).determinant();
    Scalar x33 = (SquareMatrix<Scalar, 3>((*this)(0,0), (*this)(0,1), (*this)(0,2), (*this)(1,0), (*this)(1,1), (*this)(1,2), (*this)(2,0), (*this)(2,1), (*this)(2,2))).determinant();
    return SquareMatrix<Scalar,4>(x00/det, x10/det, x20/det, x30/det, x01/det, x11/det, x21/det, x31/det, x02/det, x12/det, x22/det, x32/det, x03/det, x13/det, x23/det, x33/det);
}

template <typename Scalar>
Scalar SquareMatrix<Scalar,4>::determinant() const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    return eigen_matrix_4x4_.determinant();
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    return data_[0][0]*data_[1][1]*data_[2][2]*data_[3][3] + data_[0][0]*data_[1][2]*data_[2][3]*data_[3][1] + data_[0][0]*data_[1][3]*data_[2][1]*data_[3][2]
        - (data_[0][0]*data_[1][1]*data_[2][3]*data_[3][2] + data_[0][0]*data_[1][2]*data_[2][1]*data_[3][3] + data_[0][0]*data_[1][3]*data_[2][2]*data_[3][1])
        + data_[0][1]*data_[1][0]*data_[2][3]*data_[3][2] + data_[0][1]*data_[1][2]*data_[2][0]*data_[3][3] + data_[0][1]*data_[1][3]*data_[2][2]*data_[3][0]
        - (data_[0][1]*data_[1][0]*data_[2][2]*data_[3][3] + data_[0][1]*data_[1][2]*data_[2][3]*data_[3][0] + data_[0][1]*data_[1][3]*data_[2][0]*data_[3][2])
        + data_[0][2]*data_[1][0]*data_[2][1]*data_[3][3] + data_[0][2]*data_[1][1]*data_[2][3]*data_[3][0] + data_[0][2]*data_[1][3]*data_[2][0]*data_[3][1]
        - (data_[0][2]*data_[1][0]*data_[2][3]*data_[3][1] + data_[0][2]*data_[1][1]*data_[2][0]*data_[3][3] + data_[0][2]*data_[1][3]*data_[2][1]*data_[3][0])
        + data_[0][3]*data_[1][0]*data_[2][2]*data_[3][1] + data_[0][3]*data_[1][1]*data_[2][0]*data_[3][2] + data_[0][3]*data_[1][2]*data_[2][1]*data_[3][0]
        - (data_[0][3]*data_[1][0]*data_[2][1]*data_[3][2] + data_[0][3]*data_[1][1]*data_[2][2]*data_[3][0] + data_[0][3]*data_[1][2]*data_[2][0]*data_[3][1]);
#endif
}

template <typename Scalar>
Scalar SquareMatrix<Scalar,4>::trace() const
{
    return (*this)(0,0) + (*this)(1,1) + (*this)(2,2) + (*this)(3,3);
}

template <typename Scalar>
Scalar SquareMatrix<Scalar,4>::doubleContraction(const SquareMatrix<Scalar,4> &mat2) const
{
    Scalar result = 0;
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j < 4; ++j)
            result += (*this)(i,j)*mat2(i,j);
    return result;
}

template <typename Scalar>
Scalar SquareMatrix<Scalar,4>::frobeniusNorm() const
{
    Scalar result = 0;
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j < 4; ++j)
            result += (*this)(i,j)*(*this)(i,j);
    return sqrt(result);
}

template <typename Scalar>
SquareMatrix<Scalar,4> SquareMatrix<Scalar,4>::identityMatrix()
{
    return SquareMatrix<Scalar,4>(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
}

template <typename Scalar>
void SquareMatrix<Scalar,4>::singularValueDecomposition(SquareMatrix<Scalar,4> &left_singular_vectors,
                                                        Vector<Scalar,4> &singular_values,
                                                        SquareMatrix<Scalar,4> &right_singular_vectors) const
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    //hack: Eigen::SVD does not support integer types, hence we cast Scalar to long double for decomposition
    Eigen::Matrix<long double,4,4> temp_matrix;
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j < 4; ++j)
            temp_matrix(i,j) = static_cast<long double>(eigen_matrix_4x4_(i,j));
    Eigen::JacobiSVD<Eigen::Matrix<long double,4,4> > svd(temp_matrix,Eigen::ComputeThinU|Eigen::ComputeThinV);
    const Eigen::Matrix<long double,4,4> &left = svd.matrixU(), &right = svd.matrixV();
    const Eigen::Matrix<long double,4,1> &values = svd.singularValues();
    for(unsigned int i = 0; i < 4; ++i)
    {
        singular_values[i] = static_cast<Scalar>(values(i,0));
        for(unsigned int j = 0; j < 4; ++j)
        {
            left_singular_vectors(i,j) = static_cast<Scalar>(left(i,j));
            right_singular_vectors(i,j) = static_cast<Scalar>(right(i,j));
        }
    }
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    throw PhysikaException("SVD not implemeted for built in matrix!");
#endif
}

template <typename Scalar>
void SquareMatrix<Scalar,4>::singularValueDecomposition(SquareMatrix<Scalar,4> &left_singular_vectors,
                                                        SquareMatrix<Scalar,4> &singular_values_diagonal,
                                                        SquareMatrix<Scalar,4> &right_singular_vectors) const
{
    Vector<Scalar,4> singular_values;
    singularValueDecomposition(left_singular_vectors,singular_values,right_singular_vectors);
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j < 4; ++j)
            singular_values_diagonal(i,j) = (i==j) ? singular_values[i] : 0;
}

template <typename Scalar>
void SquareMatrix<Scalar,4>::eigenDecomposition(Vector<Scalar,4> &eigen_values_real, Vector<Scalar,4> &eigen_values_imag,
                                                SquareMatrix<Scalar,4> &eigen_vectors_real, SquareMatrix<Scalar,4> &eigen_vectors_imag)
{
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    //hack: Eigen::SVD does not support integer types, hence we cast Scalar to long double for decomposition
    Eigen::Matrix<long double,4,4> temp_matrix;
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j < 4; ++j)
            temp_matrix(i,j) = static_cast<long double>(eigen_matrix_4x4_(i,j));
    Eigen::EigenSolver<Eigen::Matrix<long double,4,4> > eigen(temp_matrix);
    Eigen::Matrix<std::complex<long double>,4,4> vectors = eigen.eigenvectors();
    const Eigen::Matrix<std::complex<long double>,4,1> &values = eigen.eigenvalues();
    for(unsigned int i = 0; i < 4; ++i)
    {
        eigen_values_real[i] = static_cast<Scalar>(values(i,0).real());
        eigen_values_imag[i] = static_cast<Scalar>(values(i,0).imag());
        for(unsigned int j = 0; j < 4; ++j)
        {
            eigen_vectors_real(i,j) = static_cast<Scalar>(vectors(i,j).real());
            eigen_vectors_imag(i,j) = static_cast<Scalar>(vectors(i,j).imag());
        }
    }
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    throw PhysikaException("Eigen decomposition not implemeted for built in matrix!");
#endif
}

//explicit instantiation of template so that it could be compiled into a lib
template class SquareMatrix<unsigned char,4>;
template class SquareMatrix<unsigned short,4>;
template class SquareMatrix<unsigned int,4>;
template class SquareMatrix<unsigned long,4>;
template class SquareMatrix<unsigned long long,4>;
template class SquareMatrix<signed char,4>;
template class SquareMatrix<short,4>;
template class SquareMatrix<int,4>;
template class SquareMatrix<long,4>;
template class SquareMatrix<long long,4>;
template class SquareMatrix<float,4>;
template class SquareMatrix<double,4>;
template class SquareMatrix<long double,4>;

}  //end of namespace Physika
