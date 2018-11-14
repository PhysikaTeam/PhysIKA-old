/*
 * @file generalized_matrix.cpp
 * @brief generalized matrix class to represent the coefficient matrix A in a linear system
 *        Ax = b. The generalized matrix hides sparse/dense information of A to provide a
 *        unified interface for LinearSystem class.
 * @author Fei Zhu
 *
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0.
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Core/Matrices/matrix_MxN.h"
#include "Physika_Core/Matrices/sparse_matrix.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Numerics/Linear_System_Solvers/generalized_matrix.h"

namespace Physika{

template <typename Scalar>
GeneralizedMatrix<Scalar>::GeneralizedMatrix(const MatrixMxN<Scalar> &matrix)
    :dense_matrix_(NULL),sparse_matrix_(NULL)
{
    dense_matrix_ = new MatrixMxN<Scalar>(matrix);
    PHYSIKA_ASSERT(dense_matrix_);
}

template <typename Scalar>
GeneralizedMatrix<Scalar>::GeneralizedMatrix(const SparseMatrix<Scalar> &matrix)
    :dense_matrix_(NULL),sparse_matrix_(NULL)
{
    sparse_matrix_ = new SparseMatrix<Scalar>(matrix);
    PHYSIKA_ASSERT(sparse_matrix_);
}

template <typename Scalar>
GeneralizedMatrix<Scalar>::GeneralizedMatrix(const GeneralizedMatrix<Scalar> &matrix)
{
    if(matrix.dense_matrix_)
    {
        this->dense_matrix_ = new MatrixMxN<Scalar>(*(matrix.dense_matrix_));
        PHYSIKA_ASSERT(this->dense_matrix_);
    }
    else
        this->dense_matrix_ = NULL;
    if(matrix.sparse_matrix_)
    {
        this->sparse_matrix_ = new SparseMatrix<Scalar>(*(matrix.sparse_matrix_));
        PHYSIKA_ASSERT(this->sparse_matrix_);
    }
    else
        this->sparse_matrix_ = NULL;
}

template <typename Scalar>
GeneralizedMatrix<Scalar>::~GeneralizedMatrix()
{
    clearMatrix();
}

template <typename Scalar>
unsigned int GeneralizedMatrix<Scalar>::rows() const
{
    if(dense_matrix_)
        return dense_matrix_->rows();
    if(sparse_matrix_)
        return sparse_matrix_->rows();
    return 0;
}

template <typename Scalar>
unsigned int GeneralizedMatrix<Scalar>::cols() const
{
    if(dense_matrix_)
        return dense_matrix_->cols();
    if(sparse_matrix_)
        return sparse_matrix_->cols();
    return 0;
}

template <typename Scalar>
GeneralizedMatrix<Scalar>& GeneralizedMatrix<Scalar>::operator= (const GeneralizedMatrix<Scalar> &matrix)
{
    clearMatrix();
    if(matrix.dense_matrix_)
        dense_matrix_ = new MatrixMxN<Scalar>(*(matrix.dense_matrix_));
    if(matrix.sparse_matrix_)
        sparse_matrix_ = new SparseMatrix<Scalar>(*(matrix.sparse_matrix_));
    return *this;
}

template <typename Scalar>
bool GeneralizedMatrix<Scalar>::operator== (const GeneralizedMatrix<Scalar> &matrix) const
{
    return (*dense_matrix_ == *(matrix.dense_matrix_)) && (*sparse_matrix_ == *(matrix.sparse_matrix_));
}

template <typename Scalar>
bool GeneralizedMatrix<Scalar>::operator!= (const GeneralizedMatrix<Scalar> &matrix) const
{
    return !(*this == matrix);
}

template <typename Scalar>
const MatrixMxN<Scalar>* GeneralizedMatrix<Scalar>::denseMatrix() const
{
    return dense_matrix_;
}

template <typename Scalar>
MatrixMxN<Scalar>* GeneralizedMatrix<Scalar>::denseMatrix()
{
    return dense_matrix_;
}

template <typename Scalar>
const SparseMatrix<Scalar>* GeneralizedMatrix<Scalar>::sparseMatrix() const
{
    return sparse_matrix_;
}

template <typename Scalar>
SparseMatrix<Scalar>* GeneralizedMatrix<Scalar>::sparseMatrix()
{
    return sparse_matrix_;
}

template <typename Scalar>
void GeneralizedMatrix<Scalar>::clearMatrix()
{
    if(dense_matrix_)
        delete dense_matrix_;
    if(sparse_matrix_)
        delete sparse_matrix_;
    dense_matrix_ = NULL;
    sparse_matrix_ = NULL;
}

//explicit instantiations
template class GeneralizedMatrix<float>;
template class GeneralizedMatrix<double>;

}  //end of namespace Physika
