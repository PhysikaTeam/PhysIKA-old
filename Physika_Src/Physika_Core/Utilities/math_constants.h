/*
 * @file math_constants.h 
 * @brief This file is used to define math constants frequently used in Physika.
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

#ifndef PHYSIKA_CORE_UTILITIES_MATH_CONSTANTS_H_
#define PHYSIKA_CORE_UTILITIES_MATH_CONSTANTS_H_

#include <limits>

namespace Physika{

const double PI = 3.14159265358979323846;
const double E = 2.71828182845904523536;
const float FLOAT_EPSILON = std::numeric_limits<float>::epsilon();
const double DOUBLE_EPSILON = std::numeric_limits<double>::epsilon();

}

#endif //PHYSIKA_CORE_UTILITIES_MATH_CONSTANTS_H_
