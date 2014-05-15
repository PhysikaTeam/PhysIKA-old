/*
 * @file sparse_matrix.cpp 
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

#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Matrices/sparse_matrix.h"

namespace Physika{

template <typename Scalar>
SparseMatrix<Scalar>::SparseMatrix()
{
    allocMemory(0,0);
}

template <typename Scalar>
SparseMatrix<Scalar>::SparseMatrix(int rows, int cols)
{
    PHYSIKA_ASSERT(rows>=0&&cols>=0);
    allocMemory(rows,cols);
}

template <typename Scalar>
SparseMatrix<Scalar>::SparseMatrix(const SparseMatrix<Scalar> &mat2)
{
    allocMemory(mat2.rows(),mat2.cols());
    *this = mat2;
}

template <typename Scalar>
void SparseMatrix<Scalar>::allocMemory(int rows, int cols)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    rows_ = rows;
    cols_ = cols;
    row_length_.resize(rows_);
    column_indices_.resize(rows_);
    column_entries_.resize(rows_);
    for(int i = 0; i < rows_; ++i)
    {
	column_indices_[i].resize(default_chunk_size_);
	column_entries_[i].resize(default_chunk_size_);
	column_indices_[i].zero();
	column_entries_[i].zero();
    }
#endif
}

template <typename Scalar>
SparseMatrix<Scalar>::~SparseMatrix()
{
}

template <typename Scalar>
int SparseMatrix<Scalar>::rows() const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    return rows_;
#endif
}

template <typename Scalar>
int SparseMatrix<Scalar>::cols() const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    return cols_;
#endif
}

template <typename Scalar>
int SparseMatrix<Scalar>::nonZeros() const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    int sum = 0;
    for(int i = 0 ; i < rows_; ++i)
        sum += row_length_[i];
    return sum;
#endif
}

template <typename Scalar>
void SparseMatrix<Scalar>::resize(int new_rows, int new_cols)
{
    PHYSIKA_ASSERT(new_rows>=0&&new_cols>=0);
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    allocMemory(new_rows,new_cols);
#endif
}

template <typename Scalar>
Scalar SparseMatrix<Scalar>::operator() (int i, int j) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    PHYSIKA_ASSERT(i>=0&&i<rows_);
    PHYSIKA_ASSERT(j>=0&&j<cols_);
    for(int idx = 0; idx < row_length_[i]; ++idx)
    {
        if(column_indices_[i][idx] == j)
            return column_entries_[i][idx];
    }
    return 0;//if is not non-zero entry, return 0
#endif
}

template <typename Scalar>
void SparseMatrix<Scalar>::setEntry(int i, int j, Scalar value)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    PHYSIKA_ASSERT(i>=0&&i<rows_);
    PHYSIKA_ASSERT(j>=0&&j<cols_);
    bool existing_entry = false;
    for(int idx = 0; idx < row_length_[i]; ++idx)
    {
        if(column_indices_[i][idx] == j)
        {
            column_entries_[i][idx] = value;
            existing_entry = true;
        }
    }
    if(!existing_entry)
    {
        int cur_entry_in_row = row_length_[i];
        int cur_alloc_size = column_indices_[i].elementCount();
        if(cur_entry_in_row >= cur_alloc_size)
        {

        }
    }
#endif
}

template <typename Scalar>
SparseMatrix<Scalar> SparseMatrix<Scalar>::operator+ (const SparseMatrix<Scalar> &mat2) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
#endif
}

template <typename Scalar>
SparseMatrix<Scalar>& SparseMatrix<Scalar>::operator+= (const SparseMatrix<Scalar> &mat2)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
#endif
}

template <typename Scalar>
SparseMatrix<Scalar> SparseMatrix<Scalar>::operator- (const SparseMatrix<Scalar> &mat2) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
#endif
}

template <typename Scalar>
SparseMatrix<Scalar>& SparseMatrix<Scalar>::operator-= (const SparseMatrix<Scalar> &mat2)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
#endif
}

template <typename Scalar>
SparseMatrix<Scalar>& SparseMatrix<Scalar>::operator= (const SparseMatrix<Scalar> &mat2)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
#endif
}

template <typename Scalar>
bool SparseMatrix<Scalar>::operator== (const SparseMatrix<Scalar> &mat2) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
#endif
}

template <typename Scalar>
SparseMatrix<Scalar> SparseMatrix<Scalar>::operator* (Scalar scale) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
#endif
}

template <typename Scalar>
SparseMatrix<Scalar>& SparseMatrix<Scalar>::operator*=(Scalar scale)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
#endif
}

template <typename Scalar>
SparseMatrix<Scalar> SparseMatrix<Scalar>::operator/ (Scalar scale) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
#endif
}

template <typename Scalar>
SparseMatrix<Scalar>& SparseMatrix<Scalar>::operator/=(Scalar scale)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
#endif
}

}  //end of namespace Physika
