/*
 * @file glm_utilities.h
 * @brief utility functions for glm math library
 * @author FeiZhu
 *
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0.
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#pragma once

#include <glm/glm.hpp>
#include "Core/Vectors/vector_3d.h"
//#include "Rendering/Color/color.h"

namespace PhysIKA {

// inline glm::vec3 convertCol3(const Color4f & col)
// {
//     return glm::vec3(col.redChannel(), col.greenChannel(), col.blueChannel());
// }

template <typename Scalar>
inline glm::vec3 convert(const Vector<Scalar, 3>& val)
{
    return glm::vec3(val[0], val[1], val[2]);
}

}  //end of namespace PhysIKA
