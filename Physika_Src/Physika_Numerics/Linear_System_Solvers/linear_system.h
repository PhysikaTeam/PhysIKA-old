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
 * Derive the class by implementing the multiply method if matrix A is not explicitly provided
 */
template <typename Scalar>
class LinearSystem
{
public:
    LinearSystem(); //construct without coefficent matrix provided
    explicit LinearSystem(const GeneralizedMatrix<Scalar> &coefficient_matrix); //construct with coefficient matrix explicitly provided
    virtual ~LinearSystem();

    //get the coefficient matrix,return NULL if the specific implementation of linear system did not provide one
    const GeneralizedMatrix<Scalar>* coefficientMatrix() const;
    GeneralizedMatrix<Scalar>* coefficientMatrix();
    //set the coefficient matrix
    void setCoefficientMatrix(const GeneralizedMatrix<Scalar> &matrix);

    //the method for iterative solvers so that matrix A does not need to be explicitly provided
    //input x return Ax
    //the default implementation is multiply between matrix and plain vector
    virtual void multiply(const GeneralizedVector<Scalar> &x, GeneralizedVector<Scalar> &result) const;
protected:
    GeneralizedMatrix<Scalar> *coefficient_matrix_; //A
};

} //end of namespace Physika

#endif //PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_LINEAR_SYSTEM_H_
