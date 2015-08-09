/*
 * @file linear_system.cpp
 * @brief base class of all classes describing the linear system Ax = b. It either provides the coefficient
 *        matrix A explicitly for direct solvers or provides multiply() method for iterative solvers.
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
#include "Physika_Core/Vectors/vector_Nd.h"
#include "Physika_Core/Matrices/matrix_MxN.h"
#include "Physika_Core/Matrices/sparse_matrix.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Numerics/Linear_System_Solvers/generalized_vector.h"
#include "Physika_Numerics/Linear_System_Solvers/plain_generalized_vector.h"
#include "Physika_Numerics/Linear_System_Solvers/generalized_matrix.h"
#include "Physika_Numerics/Linear_System_Solvers/linear_system.h"

namespace Physika{

template <typename Scalar>
LinearSystem<Scalar>::LinearSystem()
:coefficient_matrix_(NULL)
{

}

template <typename Scalar>
LinearSystem<Scalar>::LinearSystem(const GeneralizedMatrix<Scalar> &coefficient_matrix)
:coefficient_matrix_(NULL)
{
    coefficient_matrix_ = new GeneralizedMatrix<Scalar>(coefficient_matrix);
}

template <typename Scalar>
LinearSystem<Scalar>::~LinearSystem()
{
    if(coefficient_matrix_)
        delete coefficient_matrix_;
}

template <typename Scalar>
const GeneralizedMatrix<Scalar>* LinearSystem<Scalar>::coefficientMatrix() const
{
    return coefficient_matrix_;
}

template <typename Scalar>
GeneralizedMatrix<Scalar>* LinearSystem<Scalar>::coefficientMatrix()
{
    return coefficient_matrix_;
}

template <typename Scalar>
void LinearSystem<Scalar>::setCoefficientMatrix(const GeneralizedMatrix<Scalar> &matrix)
{
    if(coefficient_matrix_)
        delete coefficient_matrix_;
    coefficient_matrix_ = new GeneralizedMatrix<Scalar>(matrix);
}

template <typename Scalar>
void LinearSystem<Scalar>::multiply(const GeneralizedVector<Scalar> &x, GeneralizedVector<Scalar> &result) const
{
    if(coefficient_matrix_)
    {
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
    else
        throw PhysikaException("Coeffcient matrix not provided!");
}

//explicit instantiations
template class LinearSystem<float>;
template class LinearSystem<double>;

}  //end of namespace Physika
