/*
 * @file iterative_solver.cpp
 * @brief base class of all iterative solvers to solve linear system Ax = b
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

#include <limits>
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Numerics/Linear_System_Solvers/iterative_solver.h"

namespace Physika{

template <typename Scalar>
IterativeSolver<Scalar>::IterativeSolver()
    :LinearSystemSolver<Scalar>(),
    tolerance_(1.0e-6),max_iterations_(1000),
    residual_magnitude_sqr_(0), iterations_used_(0), status_log_(false)
{
    residual_magnitude_sqr_ = (std::numeric_limits<Scalar>::max)();
}

template <typename Scalar>
IterativeSolver<Scalar>::~IterativeSolver()
{

}

template <typename Scalar>
Scalar IterativeSolver<Scalar>::tolerance() const
{
    return tolerance_;
}

template <typename Scalar>
void IterativeSolver<Scalar>::setTolerance(Scalar tol)
{
    tolerance_ = tol;
}

template <typename Scalar>
unsigned int IterativeSolver<Scalar>::maxIterations() const
{
    return max_iterations_;
}

template <typename Scalar>
void IterativeSolver<Scalar>::setMaxIterations(unsigned int iter)
{
    max_iterations_ = iter;
}

template <typename Scalar>
Scalar IterativeSolver<Scalar>::residualMagnitude() const
{
    return sqrt(residual_magnitude_sqr_);
}

template <typename Scalar>
unsigned int IterativeSolver<Scalar>::iterationsUsed() const
{
    return iterations_used_;
}

template <typename Scalar>
void IterativeSolver<Scalar>::reset()
{
    residual_magnitude_sqr_ = (std::numeric_limits<Scalar>::max)();
    iterations_used_ = 0;
}

template <typename Scalar>
void IterativeSolver<Scalar>::enableStatusLog()
{
    status_log_ = true;
}

template <typename Scalar>
void IterativeSolver<Scalar>::disableStatusLog()
{
    status_log_ = false;
}

//explicit instantiations
template class IterativeSolver<float>;
template class IterativeSolver<double>;

}  //end of namespace Physika
