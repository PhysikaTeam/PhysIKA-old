/*
 * @file linear_system_solver.cpp
 * @brief base class of all solvers to solve linear system Ax = b
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

#include "Physika_Numerics/Linear_System_Solvers/linear_system_solver.h"

namespace Physika{

template <typename Scalar>
LinearSystemSolver<Scalar>::LinearSystemSolver()
    :use_preconditioner_(false)
{

}

template <typename Scalar>
LinearSystemSolver<Scalar>::~LinearSystemSolver()
{

}

template <typename Scalar>
bool LinearSystemSolver<Scalar>::solve(const LinearSystem<Scalar> &system, const GeneralizedVector<Scalar> &b,
                                       GeneralizedVector<Scalar> &x)
{
    if(use_preconditioner_)
        return solveWithPreconditioner(system,b,x);
    else
        return solveWithoutPreconditioner(system,b,x);
}

//explicit instantiation
template class LinearSystemSolver<float>;
template class LinearSystemSolver<double>;

} //end of namespace Physika
