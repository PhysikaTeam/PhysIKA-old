/*
 * @file iterative_solver.h
 * @brief base class of all iterative solvers to solve linear system Ax = b
 * @author Fei Zhu
 *
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_ITERATIVE_SOLVER_H_
#define PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_ITERATIVE_SOLVER_H_

#include "Physika_Numerics/Linear_System_Solvers/linear_system_solver.h"

namespace Physika{

template <typename Scalar>
class IterativeSolver: public LinearSystemSolver<Scalar>
{
public:
    IterativeSolver();
    virtual ~IterativeSolver();
    virtual bool solve(const LinearSystem<Scalar> &system, const GeneralizedVector<Scalar> &b, GeneralizedVector<Scalar> &x) = 0;
    //set solver options
    Scalar tolerance() const;
    void setTolerance(Scalar tol);
    unsigned int maxIterations() const;
    void setMaxIterations(unsigned int iter);
    //query solver state
    Scalar residualMagnitude() const;
    unsigned int iterationsUsed() const;
    //reset solver
    void reset();
protected:
    Scalar tolerance_; //terminate tolerance
    unsigned int max_iterations_; //maximum iterations allowed
    Scalar residual_magnitude_sqr_; // |b-Ax|^2
    unsigned int iterations_used_;  //iterations actually used
};

}  //end of namespace Physika

#endif //PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_ITERATIVE_SOLVER_H_
