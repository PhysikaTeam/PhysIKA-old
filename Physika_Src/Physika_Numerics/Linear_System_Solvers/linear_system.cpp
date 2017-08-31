/*
 * @file linear_system.cpp
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

#include "Physika_Numerics/Linear_System_Solvers/linear_system.h"

namespace Physika{

template <typename Scalar>
LinearSystem<Scalar>::LinearSystem()
{

}

template <typename Scalar>
LinearSystem<Scalar>::~LinearSystem()
{

}

//explicit instantiations
template class LinearSystem<float>;
template class LinearSystem<double>;

}  //end of namespace Physika
