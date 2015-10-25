/*
 * @file matrix_linear_system.cpp
 * @brief base class of all linear system classes that provide coefficient matrix explicitly
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

#include <cstddef>
#include <typeinfo>
#include <iostream>
#include "Physika_Core/Vectors/vector_Nd.h"
#include "Physika_Core/Matrices/matrix_MxN.h"
#include "Physika_Core/Matrices/sparse_matrix.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Numerics/Linear_System_Solvers/generalized_vector.h"
#include "Physika_Numerics/Linear_System_Solvers/plain_generalized_vector.h"
#include "Physika_Numerics/Linear_System_Solvers/generalized_matrix.h"
#include "Physika_Numerics/Linear_System_Solvers/matrix_linear_system.h"

namespace Physika{

template <typename Scalar>
MatrixLinearSystem<Scalar>::MatrixLinearSystem()
:LinearSystem<Scalar>(),coefficient_matrix_(NULL),preconditioner_(NULL)
{

}

template <typename Scalar>
MatrixLinearSystem<Scalar>::MatrixLinearSystem(const GeneralizedMatrix<Scalar> &coefficient_matrix)
:LinearSystem<Scalar>(),coefficient_matrix_(NULL),preconditioner_(NULL)
{
    coefficient_matrix_ = new GeneralizedMatrix<Scalar>(coefficient_matrix);
}

template <typename Scalar>
MatrixLinearSystem<Scalar>::~MatrixLinearSystem()
{
    if(coefficient_matrix_)
        delete coefficient_matrix_;
    if(preconditioner_)
        delete preconditioner_;
}

template <typename Scalar>
const GeneralizedMatrix<Scalar>* MatrixLinearSystem<Scalar>::coefficientMatrix() const
{
    return coefficient_matrix_;
}

template <typename Scalar>
GeneralizedMatrix<Scalar>* MatrixLinearSystem<Scalar>::coefficientMatrix()
{
    return coefficient_matrix_;
}

template <typename Scalar>
void MatrixLinearSystem<Scalar>::setCoefficientMatrix(const GeneralizedMatrix<Scalar> &matrix)
{
    if(coefficient_matrix_)
        delete coefficient_matrix_;
    coefficient_matrix_ = new GeneralizedMatrix<Scalar>(matrix);
}

template <typename Scalar>
void MatrixLinearSystem<Scalar>::multiply(const GeneralizedVector<Scalar> &x, GeneralizedVector<Scalar> &result) const
{
    if(coefficient_matrix_)
    {
        try{
            const PlainGeneralizedVector<Scalar> &plain_x = dynamic_cast<const PlainGeneralizedVector<Scalar>&>(x);
            VectorND<Scalar> vec;
            VectorND<Scalar> raw_x = plain_x.rawVector();
            MatrixMxN<Scalar> *dense_mat = coefficient_matrix_->denseMatrix();
            if(dense_mat)
                vec = (*dense_mat)*raw_x;
            SparseMatrix<Scalar> *sparse_mat = coefficient_matrix_->sparseMatrix();
            if(sparse_mat)
                vec = (*sparse_mat)*raw_x;
            result = PlainGeneralizedVector<Scalar>(vec);
        }
        catch(std::bad_cast& e)
        {
            throw PhysikaException("Incorrect argument type!");
        }
    }
    else
        throw PhysikaException("Coefficient matrix not provided!");
}

template <typename Scalar>
const GeneralizedMatrix<Scalar>* MatrixLinearSystem<Scalar>::preconditioner() const
{
    return preconditioner_;
}

template <typename Scalar>
GeneralizedMatrix<Scalar>* MatrixLinearSystem<Scalar>::preconditioner()
{
    return preconditioner_;
}

template <typename Scalar>
void MatrixLinearSystem<Scalar>::setPreconditioner(const GeneralizedMatrix<Scalar> &matrix)
{
    if(preconditioner_)
        delete preconditioner_;
    preconditioner_ = new GeneralizedMatrix<Scalar>(matrix);
}

template <typename Scalar>
void MatrixLinearSystem<Scalar>::preconditionerMultiply(const GeneralizedVector<Scalar> &x, GeneralizedVector<Scalar> &result) const
{
    if(preconditioner_)
    {
        try{
            const PlainGeneralizedVector<Scalar> &plain_x = dynamic_cast<const PlainGeneralizedVector<Scalar>&>(x);
            VectorND<Scalar> vec;
            VectorND<Scalar> raw_x = plain_x.rawVector();
            MatrixMxN<Scalar> *dense_mat = preconditioner_->denseMatrix();
            if(dense_mat)
                vec = (*dense_mat)*raw_x;
            SparseMatrix<Scalar> *sparse_mat = preconditioner_->sparseMatrix();
            if(sparse_mat)
                vec = (*sparse_mat)*raw_x;
            result = PlainGeneralizedVector<Scalar>(vec);
        }
        catch(std::bad_cast& e)
        {
            throw PhysikaException("Incorrect argument type!");
        }
    }
    else
        throw PhysikaException("Preconditioner not provided!");
}

template <typename Scalar>
void MatrixLinearSystem<Scalar>::computeJacobiPreconditioner()
{//jacobi preconditioner: diagonal matrix with entries of 1.0/A(i,i)
    if(coefficient_matrix_)
    {
        if(preconditioner_)
            delete preconditioner_;
        MatrixMxN<Scalar> *dense_A = coefficient_matrix_->denseMatrix();
        if(dense_A)
        {
            MatrixMxN<Scalar> dense_T(dense_A->rows(),dense_A->cols(),0.0);
            PHYSIKA_ASSERT(dense_T.rows() == dense_T.cols());
            for(unsigned int i = 0; i < dense_T.rows(); ++i)
                dense_T(i,i) = 1.0/(*dense_A)(i,i);
            preconditioner_ = new GeneralizedMatrix<Scalar>(dense_T);
            return;
        }
        SparseMatrix<Scalar> *sparse_A = coefficient_matrix_->sparseMatrix();
        if(sparse_A)
        {
            SparseMatrix<Scalar> sparse_T(sparse_A->rows(),sparse_A->cols());
            PHYSIKA_ASSERT(sparse_T.rows() == sparse_T.cols());
            for(unsigned int i = 0; i < sparse_T.rows(); ++i)
                sparse_T.setEntry(i,i,1.0/(*sparse_A)(i,i));
            preconditioner_ = new GeneralizedMatrix<Scalar>(sparse_T);
            return;
        }
    }
    else
        std::cerr<<"Warning: coefficient matrix not explicitly provided, computeJacobiPreconditioner() operation ignored!\n";
}

//explicit instantiations
template class MatrixLinearSystem<float>;
template class MatrixLinearSystem<double>;

}  //end of namespace Physika
