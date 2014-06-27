/*
 * @file matrix_MxN.h 
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

#ifndef PHYSIKA_CORE_MATRICES_MATRIX_MXN_H_
#define PHYSIKA_CORE_MATRICES_MATRIX_MXN_H_

#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/type_utilities.h"
#include "Physika_Core/Vectors/vector_Nd.h"
#include "Physika_Core/Matrices/matrix_base.h"

namespace Physika{

/*
 * MatrixMxN<Scalar> are defined for C++ fundamental integer types and floating-point types
 */

template <typename Scalar> 
class MatrixMxN: public MatrixBase
{
public:
    MatrixMxN(); //construct an empty matrix
    MatrixMxN(int rows, int cols); //construct an uninitialized matrix of size rows*cols                  
    MatrixMxN(int rows, int cols, Scalar *entries);  //construct a matrix with given size and data
    MatrixMxN(const MatrixMxN<Scalar>&);  //copy constructor
    ~MatrixMxN();
    int rows() const;
    int cols() const;
    void resize(int new_rows, int new_cols);  //resize the matrix to new_rows*new_cols
    Scalar& operator() (int i, int j);
    const Scalar& operator() (int i, int j) const;
    MatrixMxN<Scalar> operator+ (const MatrixMxN<Scalar> &) const;
    MatrixMxN<Scalar>& operator+= (const MatrixMxN<Scalar> &);
    MatrixMxN<Scalar> operator- (const MatrixMxN<Scalar> &) const;
    MatrixMxN<Scalar>& operator-= (const MatrixMxN<Scalar> &);
    MatrixMxN<Scalar>& operator= (const MatrixMxN<Scalar> &);
    bool operator== (const MatrixMxN<Scalar> &)const;
    MatrixMxN<Scalar> operator* (Scalar) const;
    MatrixMxN<Scalar>& operator*= (Scalar);
    VectorND<Scalar> operator* (const VectorND<Scalar> &) const;
    MatrixMxN<Scalar> operator/ (Scalar) const;
    MatrixMxN<Scalar>& operator/= (Scalar);
    MatrixMxN<Scalar> transpose() const;
    MatrixMxN<Scalar> inverse() const;
    MatrixMxN<Scalar> cofactorMatrix() const;
    Scalar determinant() const;
    Scalar trace() const;
    Scalar doubleContraction(const MatrixMxN<Scalar> &) const;
protected:
    void allocMemory(int rows, int cols);
protected:
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> *ptr_eigen_matrix_MxN_;
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    Scalar *data_;
    int rows_,cols_;
#endif
protected:
    PHYSIKA_STATIC_ASSERT((is_integer<Scalar>::value||is_floating_point<Scalar>::value),
                      "MatrixMxN<Scalar> are only defined for integer types and floating-point types.");
};

//overriding << for MatrixMxN
template <typename Scalar>
std::ostream& operator<< (std::ostream &s, const MatrixMxN<Scalar> &mat)
{
    s<<"[";
    for(int i = 0; i < mat.rows(); ++i)
    {
        for(int j = 0; j < mat.cols() - 1; ++j)
	    s<<mat(i,j)<<", ";
	if(i != mat.rows()-1)
	    s<<mat(i,mat.cols() - 1)<<"; ";
    }
    s<<"]";
    return s;
}

//make * operator commutative
template <typename S, typename T>
MatrixMxN<T> operator* (S scale, const MatrixMxN<T> &mat)
{
    return mat*scale;
}

//convenient typedefs
typedef MatrixMxN<float> MatrixXf;
typedef MatrixMxN<double> MatrixXd;
typedef MatrixMxN<int> MatrixXi;

}

#endif //PHYSIKA_CORE_MATRICES_MATRIX_MXN_H_

