/*
 * @file math_constants.h 
 * @brief This file is used to define miscellaneous (except math-related) constants frequently used in Physika.
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
#ifndef PHYSIKA_CORE_UTILITIES_MISC_CONSTANTS_H_
#define PHYSIKA_CORE_UTILITIES_MISC_CONSTANTS_H_

namespace Physika{
  const int Dynamic = -1;  //This value means that a quantity is not known at compile-time
  const unsigned int RowMajorBit = 0x1;  //For a matrix, this means that the storage order is row-major. If this bit is not set, the storage order is column-major.
}

#endif //PHYSIKA_CORE_UTILITIES_MISC_CONSTANTS_H_
