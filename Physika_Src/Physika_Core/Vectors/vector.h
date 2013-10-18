/*
 * @file vector.h 
 * @This file contains all vector-related header files and common used typedefs. 
 * @author Sheng Yang
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_VECTORS_VECTOR_H_
#define PHYSIKA_CORE_VECTORS_VECTOR_H_

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

namespace Type{

typedef Vector2D<float> Vector2f;
typedef Vector2D<double> Vector2d;

typedef Vector3D<float> Vector3f;
typedef Vector3D<double> Vector3d;
}  //end of namespace TYPE

}  //end of namespace Physika

#endif //PHYSIKA_CORE_VECTORS_VECTOR_H_
