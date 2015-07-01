/*
 * @file dimension_trait.h 
 * @Trait class for trait-based TMP.
 *  Help to overload different functions for different dimensions during compiling.
 *  See "Effective C++ (3rd Edition)", section 47: "use traits classes for information about types" to learn more.
 * @author Tianxiang Zhang, Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_UTILITIES_DIMENSION_TRAIT_H_
#define PHYSIKA_CORE_UTILITIES_DIMENSION_TRAIT_H_

namespace Physika{

/*
 * Example usage:
 * Suppose source code of templated method f() is different
 * when dimension is different, in order that f() passes
 * compilation, dimension trait could be used
 *
 * template <int Dim>
 * f()
 * {
 *     DimensionTrait<Dim> trait;
 *     g(trait);   //real stuff
 * }
 *
 * g(DimensionTrait<2> trait)
 * {
 *  //2D code
 * }
 *
 * g(DimensionTrait<3> trait)
 * {
 *  //3D code
 * }
 */

template <int Dim>
struct DimensionTrait
{
};

}

#endif //PHYSIKA_CORE_UTILITIES_DIMENSION_TRAIT_H_
