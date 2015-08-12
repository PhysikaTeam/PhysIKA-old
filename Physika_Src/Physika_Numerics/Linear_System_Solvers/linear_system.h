/*
 * @file linear_system.h
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

#ifndef PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_LINEAR_SYSTEM_H_
#define PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_LINEAR_SYSTEM_H_

namespace Physika{

template <typename RawScalar> class GeneralizedVector;
template <typename Scalar> class GeneralizedMatrix;

/*
 * LinearSystem:
 * Derive the class by implementing the multiply method
 * Note: if A can not be conveniently provided explicitly for specific linear system, it is recommended
 *       to hide methods related to explicit form of A in subclass by overriding
 */
template <typename Scalar>
class LinearSystem
{
public:
    LinearSystem(); //construct without coefficent matrix provided
    explicit LinearSystem(const GeneralizedMatrix<Scalar> &coefficient_matrix); //construct with coefficient matrix explicitly provided
    virtual ~LinearSystem();

    //get the coefficient matrix,return NULL if not explicitly set
    const GeneralizedMatrix<Scalar>* coefficientMatrix() const;
    GeneralizedMatrix<Scalar>* coefficientMatrix();
    //set the coefficient matrix
    void setCoefficientMatrix(const GeneralizedMatrix<Scalar> &matrix);
    //the method for iterative solvers so that matrix A does not need to be explicitly provided
    //input x return Ax
    //the default implementation is multiply between matrix and plain vector
    virtual void multiply(const GeneralizedVector<Scalar> &x, GeneralizedVector<Scalar> &result) const;

    //get the preconditioner matrix, return NULL if not explicitly set
    const GeneralizedMatrix<Scalar>* preconditioner() const;
    GeneralizedMatrix<Scalar>* preconditioner();
    //set the preconditioner
    void setPreconditioner(const GeneralizedMatrix<Scalar> &matrix);
    //the method for iterative solvers so that preconditioner needn't to be explicitly provided
    //input x return Tx
    //the default implementation here is multiply between matrix and plain vector
    virtual void preconditionerMultiply(const GeneralizedVector<Scalar> &x, GeneralizedVector<Scalar> &result) const;
    //predefined preconditioners: only work if A is explicitly provided
    void computeJacobiPreconditioner(); //aka diagonal preconditioner
protected:
    GeneralizedMatrix<Scalar> *coefficient_matrix_; //A
    GeneralizedMatrix<Scalar> *preconditioner_; //preconditioner of the linear system
};

} //end of namespace Physika

#endif //PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_LINEAR_SYSTEM_H_
