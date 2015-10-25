/*
 * @file matrix_linear_system.h
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

#ifndef PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_MATRIX_LINEAR_SYSTEM_H_
#define PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_MATRIX_LINEAR_SYSTEM_H_

#include "Physika_Numerics/Linear_System_Solvers/linear_system.h"

namespace Physika{

template <typename RawScalar> class GeneralizedVector;
template <typename Scalar> class GeneralizedMatrix;

template <typename Scalar>
class MatrixLinearSystem: public LinearSystem<Scalar>
{
public:
    MatrixLinearSystem(); //construct without coefficient matrix provided
    //construct with coefficient matrix explicitly provided
    explicit MatrixLinearSystem(const GeneralizedMatrix<Scalar> &coefficient_matrix);
    virtual ~MatrixLinearSystem();

    //get the coefficient matrix,return NULL if not explicitly set
    const GeneralizedMatrix<Scalar>* coefficientMatrix() const;
    GeneralizedMatrix<Scalar>* coefficientMatrix();
    //set the coefficient matrix
    void setCoefficientMatrix(const GeneralizedMatrix<Scalar> &matrix);
    //the default implementation is multiply between matrix and PlainGeneralizedVector
    virtual void multiply(const GeneralizedVector<Scalar> &x, GeneralizedVector<Scalar> &result) const;

    //get the preconditioner matrix, return NULL if not explicitly set
    const GeneralizedMatrix<Scalar>* preconditioner() const;
    GeneralizedMatrix<Scalar>* preconditioner();
    //set the preconditioner
    void setPreconditioner(const GeneralizedMatrix<Scalar> &matrix);
    //the default implementation here is multiply between matrix and PlainGeneralizedVector
    virtual void preconditionerMultiply(const GeneralizedVector<Scalar> &x, GeneralizedVector<Scalar> &result) const;
    //predefined preconditioners: only work if A is explicitly provided
    void computeJacobiPreconditioner(); //aka diagonal preconditioner
protected:
    //disable default copy
    MatrixLinearSystem(const MatrixLinearSystem<Scalar> &linear_system);
    MatrixLinearSystem<Scalar>& operator= (const MatrixLinearSystem<Scalar> &linear_system);
protected:
    GeneralizedMatrix<Scalar> *coefficient_matrix_; //A
    GeneralizedMatrix<Scalar> *preconditioner_; //preconditioner of the linear system
};

} //end of namespace Physika

#endif //PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_MATRIX_LINEAR_SYSTEM_H_
