/*
 * @file conjugate_gradient_solver.cpp
 * @brief implementation of conjugate gradient solver for semi-definite sparse linear system Ax = b
 * @reference <An Introduction to the Conjugate Gradient Method without the Agonizing Pain>
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

#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Numerics/Linear_System_Solvers/generalized_vector.h"
#include "Physika_Numerics/Linear_System_Solvers/linear_system.h"
#include "Physika_Numerics/Linear_System_Solvers/conjugate_gradient_solver.h"

namespace Physika{

template <typename Scalar>
ConjugateGradientSolver<Scalar>::ConjugateGradientSolver()
:IterativeSolver<Scalar>()
{

}

template <typename Scalar>
ConjugateGradientSolver<Scalar>::~ConjugateGradientSolver()
{

}

template <typename Scalar>
bool ConjugateGradientSolver<Scalar>::solve(const LinearSystem<Scalar> &system,
                                            const GeneralizedVector<Scalar> &b,
                                            GeneralizedVector<Scalar> &x)
{
    //see b2. of the reference for notations
    this->iterations_used_ = 0;
    Scalar tolerance_sqr = (this->tolerance_)*(this->tolerance_);
    GeneralizedVector<Scalar> *r = b.clone(); //create a vector with the same type with b
    system.multiply(x,*r); //r = Ax
    (*r) *= -1;
    (*r) += b; //r = b - Ax
    GeneralizedVector<Scalar> *d = r->clone();
    Scalar delta_0 = r->normSquared();
    Scalar delta = delta_0;
    GeneralizedVector<Scalar> *q = d->clone();
    GeneralizedVector<Scalar> *temp = d->clone();
    while((this->iterations_used_ < this->max_iterations_) && (delta > tolerance_sqr*delta_0))
    {
        system.multiply(*d,*q);
        Scalar alpha = delta/(d->dot(*q));
        *temp = *d;
        *temp *= alpha;
        x += *temp; //x = x + alpha*d
        if((this->iterations_used_)%50 == 0) //direct compute residual every 50 iterations
        {
            system.multiply(x,*r);
            (*r) *= -1;
            (*r) += b; //r = b - Ax
        }
        else
        {
            *temp = *q;
            *temp *= alpha;
            *r -= *temp; //r = r - alpha*q
        }
        Scalar delta_old = delta;
        delta = r->normSquared();
        Scalar beta = delta/delta_old;
        *temp = *d;
        *temp *= beta;
        *d = *r;
        *d += *temp;  //d = r + beta*d
        this->iterations_used_ = this->iterations_used_ + 1;
        this->residual_magnitude_sqr_ = delta;
    }
    delete r;
    delete d;
    delete q;
    delete temp;
    //return false if didn't converge with maximum iterations
    return delta > tolerance_sqr*delta_0 ? false : true;
}

//explicit instantiations
template class ConjugateGradientSolver<float>;
template class ConjugateGradientSolver<double>;

}  //end of namespace Physika
