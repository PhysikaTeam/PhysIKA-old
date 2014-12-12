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
#include "Physika_Core/Matrices/matrix_base.h"

namespace Physika{

template <typename Scalar> class VectorND;

/*
 * MatrixMxN<Scalar> are defined for C++ fundamental integer types and floating-point types
 */

template <typename Scalar> 
class MatrixMxN: public MatrixBase
{
public:
    MatrixMxN(); //construct an empty matrix
    MatrixMxN(unsigned int rows, unsigned int cols); //construct an uninitialized matrix of size rows*cols
    MatrixMxN(unsigned int rows, unsigned int cols, Scalar value); //construct an matrix of size rows*cols with the entry value                  
    MatrixMxN(unsigned int rows, unsigned int cols, Scalar *entries);  //construct a matrix with given size and data
    MatrixMxN(const MatrixMxN<Scalar>&);  //copy constructor
    ~MatrixMxN();
    unsigned int rows() const;
    unsigned int cols() const;
    void resize(unsigned int new_rows, unsigned int new_cols);  //resize the matrix to new_rows*new_cols
    Scalar& operator() (unsigned int i, unsigned int j);
    const Scalar& operator() (unsigned int i, unsigned int j) const;
    MatrixMxN<Scalar> operator+ (const MatrixMxN<Scalar> &) const;
    MatrixMxN<Scalar>& operator+= (const MatrixMxN<Scalar> &);
    MatrixMxN<Scalar> operator- (const MatrixMxN<Scalar> &) const;
    MatrixMxN<Scalar>& operator-= (const MatrixMxN<Scalar> &);
    MatrixMxN<Scalar>& operator= (const MatrixMxN<Scalar> &);
    bool operator== (const MatrixMxN<Scalar> &)const;
    bool operator!= (const MatrixMxN<Scalar> &)const;
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
    void singularValueDecomposition(MatrixMxN<Scalar> &left_singular_vectors,
                                    VectorND<Scalar> &singular_values,
                                    MatrixMxN<Scalar> &right_singular_vectors) const;
    void eigenDecomposition(VectorND<Scalar> &eigen_values_real, VectorND<Scalar> &eigen_values_imag,
                            MatrixMxN<Scalar> &eigen_vectors_real, MatrixMxN<Scalar> &eigen_vectors_imag);
protected:
    void allocMemory(unsigned int rows, unsigned int cols);
protected:
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> *ptr_eigen_matrix_MxN_;
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    Scalar *data_;
    int rows_,cols_;
#endif
private:
    void compileTimeCheck()
    {
        //MatrixMxN<Scalar> is only defined for element type of integers and floating-point types
        //compile time check
        PHYSIKA_STATIC_ASSERT((is_integer<Scalar>::value||is_floating_point<Scalar>::value),
                              "MatrixMxN<Scalar> are only defined for integer types and floating-point types.");
    }
};

//overriding << for MatrixMxN
template <typename Scalar>
inline std::ostream& operator<< (std::ostream &s, const MatrixMxN<Scalar> &mat)
{
    s<<"[";
    for(int i = 0; i < mat.rows(); ++i)
    {
        for(int j = 0; j < mat.cols()-1; ++j)
            s<<mat(i,j)<<", ";
        s<<mat(i,mat.cols()-1);
        if(i != mat.rows()-1)
            s<<"; ";
    }
    s<<"]";
    return s;
}

//make * operator commutative
template <typename S, typename T>
inline MatrixMxN<T> operator* (S scale, const MatrixMxN<T> &mat)
{
    return mat*scale;
}

//convenient typedefs
typedef MatrixMxN<float> MatrixXf;
typedef MatrixMxN<double> MatrixXd;
typedef MatrixMxN<int> MatrixXi;

}

#endif //PHYSIKA_CORE_MATRICES_MATRIX_MXN_H_
