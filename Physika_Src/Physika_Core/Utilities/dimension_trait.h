/*
 * @file dimension_trait.h 
 * @Trait class for trait-based TMP.
 *  Help to overload different functions for different dimensions during compiling.
 *  See "Effective C++ (3rd Edition)", section 47: "use traits classes for information about types" to learn more.
 * @author Tianxiang Zhang
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_UTILITIES_DIMENSION_TRAIT_H_
#define PHYSIKA_CORE_UTILITIES_DIMENSION_TRAIT_H_

namespace Physika{

template <int Dim>
struct DimensionTrait
{
};

}

#endif //PHYSIKA_CORE_UTILITIES_DIMENSION_TRAIT_H_