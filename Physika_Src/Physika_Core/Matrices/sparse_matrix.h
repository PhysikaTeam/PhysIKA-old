/*
 * @file sparse_matrix.h 
 * @brief Definition of sparse matrix, size of the matrix is dynamic.
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

#ifndef PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_H_
#define PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_H_

#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Array/array.h"
#include "Physika_Core/Matrices/matrix_base.h"

namespace Physika{

template <typename Scalar>
class SparseMatrix: public MatrixBase
{
public:
    SparseMatrix();
    SparseMatrix(int rows, int cols);
    SparseMatrix(const SparseMatrix<Scalar> &);
    ~SparseMatrix();
    int rows() const;
    int cols() const;
    int nonZeros() const;
    void resize(int new_rows, int new_cols);
    Scalar operator() (int i, int j) const;//return value of matrix entry at index (i,j). Note: cannot be used as l-value!
    void setEntry(int i, int j, Scalar value);//insert matrix entry at index (i,j), if it already exits, replace it
    SparseMatrix<Scalar> operator+ (const SparseMatrix<Scalar> &) const;
    SparseMatrix<Scalar>& operator+= (const SparseMatrix<Scalar> &);
    SparseMatrix<Scalar> operator- (const SparseMatrix<Scalar> &) const;
    SparseMatrix<Scalar>& operator-= (const SparseMatrix<Scalar> &);
    SparseMatrix<Scalar>& operator= (const SparseMatrix<Scalar> &);
    bool operator== (const SparseMatrix<Scalar> &) const;
    SparseMatrix<Scalar> operator* (Scalar) const;
    SparseMatrix<Scalar>& operator*= (Scalar);
    SparseMatrix<Scalar> operator/ (Scalar) const;
    SparseMatrix<Scalar>& operator/= (Scalar);
protected:
    void allocMemory(int rows, int cols);
protected:
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
//compressed row storage
    typedef Array<int> ArrayInt;
    typedef Array<Scalar> ArrayScalar;
    int rows_;
    int cols_;
    ArrayInt row_length_;//number of nonzero entries in each row
    Array<ArrayInt> column_indices_;//indices of columns of non-zero entries in each row
    Array<ArrayScalar> column_entries_;//values of non-zero entries in each row
    const int default_chunk_size_ = 10;//default chunk size allocated for each row
#endif
};

//overridding << for SparseMatrix<Scalar>
template <typename Scalar>
std::ostream& operator<< (std::ostream &s, const SparseMatrix<Scalar> &mat)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    for(int i = 0; i < rows_; ++i)
	for(int j = 0; j < row_length_[i]; ++j)
	{
	    int col_idx = column_indices_[i][j];
	    s<<"("<<i<<","<<col_idx<<"): "<<column_entries_[i][j]<<"\n";
        }
#endif
    return s;
}

//make * operator commutative
template <typename S, typename T>
SparseMatrix<T> operator* (S scale, const SparseMatrix<T> &mat)
{
    return mat*scale;
}

}  //end of namespace Physika

#endif //PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_H_
