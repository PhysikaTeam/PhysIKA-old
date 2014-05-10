/*
 * @file math_utilities.cpp 
 * @brief This file is used to define math constants and functions frequently used in Physika.
 * @author FeiZhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <cmath>
#include "Physika_Core/Utilities/Math_Utilities/math_utilities.h"

namespace Physika{

template <typename Scalar>
Scalar abs(Scalar value)
{
    return value>=0?value:-value;
}

template <typename Scalar>
Scalar sqrt(Scalar value)
{
    return std::sqrt(value);
}

double sqrt(int value)
{
    return std::sqrt(static_cast<double>(value));
}

}  //end of namespace Physika
