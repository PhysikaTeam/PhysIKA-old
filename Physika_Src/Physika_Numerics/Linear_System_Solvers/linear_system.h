/*
 * @file linear_system.h
 * @brief base class of all classes describing the linear system Ax = b. It provides the coefficient
 *        matrix A implicitly with multiply() method.
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

/*
 * LinearSystem:
 * Derive the class by implementing the multiply method
 *
 */
template <typename Scalar>
class LinearSystem
{
public:
    LinearSystem();
    virtual ~LinearSystem();

    //the method for iterative solvers so that matrix A does not need to be explicitly provided
    //input x return Ax
    virtual void multiply(const GeneralizedVector<Scalar> &x, GeneralizedVector<Scalar> &result) const = 0;

    //the method for iterative solvers so that preconditioner needn't to be explicitly provided
    //input x return Tx
    virtual void preconditionerMultiply(const GeneralizedVector<Scalar> &x, GeneralizedVector<Scalar> &result) const = 0;

    //the method that defines linear system specific inner-product of two vectors
    virtual Scalar innerProduct(const GeneralizedVector<Scalar> &x, const GeneralizedVector<Scalar> &y) const = 0;
protected:
    //disable default copy
    LinearSystem(const LinearSystem<Scalar> &linear_system);
    LinearSystem<Scalar>& operator= (const LinearSystem<Scalar> &linear_system);
};

} //end of namespace Physika

#endif //PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_LINEAR_SYSTEM_H_
