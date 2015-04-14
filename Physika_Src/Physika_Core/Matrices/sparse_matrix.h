/*
 * @file sparse_matrix.h 
 * @brief Definition of sparse matrix, size of the matrix is dynamic.
 * @author Fei Zhu, Liyou Xu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_H_
#define PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_H_

#include <vector>
#include <iostream>
#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/type_utilities.h"
#include "Physika_Core/Matrices/matrix_base.h"
#include "Physika_Core/Matrices/sparse_matrix_internal.h"
#include "Physika_Core/Vectors/vector_Nd.h"

namespace Physika{

template <typename Scalar> class SparseMatrixIterator;

/*
 * class SparseMatrix is a data structure used to store SparseMatrix
 * it uses the Trituple as its node in a triple list
 *
 */
template <typename Scalar>
class SparseMatrix: public MatrixBase
{
public:
    explicit SparseMatrix(SparseMatrixInternal::SparseMatrixStoreMode priority = SparseMatrixInternal::ROW_MAJOR);
	SparseMatrix(unsigned int rows, unsigned int cols, SparseMatrixInternal::SparseMatrixStoreMode priority = SparseMatrixInternal::ROW_MAJOR);
    SparseMatrix(const SparseMatrix<Scalar> &);
    ~SparseMatrix();
    unsigned int rows() const;
    unsigned int cols() const;
    //return the number of nonZero node
    unsigned int nonZeros() const;                //itinerate the whole vector once to calculate the nonzeros
    // remove a node(i,j) and adjust the orthogonal list
    bool remove(unsigned int i,unsigned int j);
    //resize the SparseMatrix and data in it will be deleted
    void resize(unsigned int new_rows, unsigned int new_cols);
    SparseMatrix<Scalar> transpose() const;
    void rowElements(unsigned int row, std::vector<Scalar> &elements) const;
    void colElements(unsigned int col, std::vector<Scalar> &elements) const;
    //return value of matrix entry at index (i,j). Note: cannot be used as l-value!
    Scalar operator() (unsigned int i, unsigned int j) const;
    //insert matrix entry at index (i,j), if it already exits, replace it
    void setEntry(unsigned int i, unsigned int j, Scalar value);
    SparseMatrix<Scalar> operator+ (const SparseMatrix<Scalar> &) const;
    SparseMatrix<Scalar>& operator+= (const SparseMatrix<Scalar> &);
    SparseMatrix<Scalar> operator- (const SparseMatrix<Scalar> &) const;
    SparseMatrix<Scalar>& operator-= (const SparseMatrix<Scalar> &);
    SparseMatrix<Scalar>& operator= (const SparseMatrix<Scalar> &);
    bool operator== (const SparseMatrix<Scalar> &) const;
    bool operator!= (const SparseMatrix<Scalar> &) const;
    SparseMatrix<Scalar> operator* (Scalar) const;
    SparseMatrix<Scalar> operator* (const SparseMatrix<Scalar> &) const;
    VectorND<Scalar> operator* (const VectorND<Scalar> &) const;
    SparseMatrix<Scalar>& operator*= (Scalar);
    SparseMatrix<Scalar> operator/ (Scalar) const;
    SparseMatrix<Scalar>& operator/= (Scalar);
    VectorND<Scalar> leftMultiplyVector(const VectorND<Scalar> &) const;
protected:
    void allocMemory(unsigned int rows, unsigned int cols, SparseMatrixInternal::SparseMatrixStoreMode priority);
protected:
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    //row-wise format or col-wise format
    unsigned int rows_;
    unsigned int cols_;
    std::vector<SparseMatrixInternal::Trituple<Scalar> > elements_; //a vector used to contain all the none-zero elements in a sparsematrix in order
    std::vector<unsigned int> line_index_;   //line_index store the index of the first non-zero element of every row when priority is equal to ROW_MAJOR 
                                             //or every col when priority is equal to COL_MAJOR
	SparseMatrixInternal::SparseMatrixStoreMode priority_;  
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
	SparseMatrixInternal::SparseMatrixStoreMode priority_;
    Eigen::SparseMatrix<Scalar> * ptr_eigen_sparse_matrix_ ;
#endif      
    friend class SparseMatrixIterator<Scalar>;  // declare friend class for iterator
private:
    void compileTimeCheck()
    {
        //SparseMatrix<Scalar> is only defined for element type of integers and floating-point types
        //compile time check
        PHYSIKA_STATIC_ASSERT((is_integer<Scalar>::value||is_floating_point<Scalar>::value),
                              "SparseMatrix<Scalar> are only defined for integer types and floating-point types.");
    }
};

//overridding << for SparseMatrix<Scalar>
template <typename Scalar>
std::ostream& operator<< (std::ostream &s, const SparseMatrix<Scalar> &mat)
{
    std::vector<SparseMatrixInternal::Trituple<Scalar> > v;
    for(unsigned int i = 0; i < mat.rows(); ++i)
    {
        v = mat.getRowElements(i);
        for(unsigned int j=0;j< v.size();++j)
            s<<" ("<< v[j].row()<<", "<<v[j].col()<<", "<<v[j].value()<<") ";
        s<<std::endl;
    }
    return s;
}

//make * operator commutative
template <typename S, typename T>
SparseMatrix<T> operator* (S scale, const SparseMatrix<T> &mat)
{
    return mat*scale;
}

//multiply a row vector with a sparse matrix
template <typename Scalar>
VectorND<Scalar> operator*(const VectorND<Scalar> &vec, const SparseMatrix<Scalar> &mat)
{
    return mat.leftMultiplyVector(vec);
}

}  //end of namespace Physika


#endif //PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_H_
