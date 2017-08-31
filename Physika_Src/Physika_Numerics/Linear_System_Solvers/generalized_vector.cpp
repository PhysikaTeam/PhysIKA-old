/*
 * @file generalized_vector.cpp
 * @brief generalized vector class to represent solution x and right hand side b in a linear system
 *        Ax = b. The generalized vector class may be derived for specific linear system for convenience.
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

#include "Physika_Numerics/Linear_System_Solvers/generalized_vector.h"

namespace Physika{

template <typename RawScalar>
GeneralizedVector<RawScalar>& GeneralizedVector<RawScalar>::operator= (const GeneralizedVector<RawScalar> &vector)
{
    copy(vector);
    return *this;
}

template class GeneralizedVector<float>;
template class GeneralizedVector<double>;

}  //end of namespace Physika
