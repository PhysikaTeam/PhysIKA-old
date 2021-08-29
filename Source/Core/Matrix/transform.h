/*
 * @file vector.h 
 * @brief This abstract class is intended to provide a uniform interface for transfrom2D and transform3D.
 *        transfrom2D and transform3D are implemented using template partial specialization of this class. 
 * @author Sheng Yang
 * 
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_TRANSFORM_TRANSFORM_H_
#define PHYSIKA_CORE_TRANSFORM_TRANSFORM_H_

#include "Core/Utility.h"

namespace PhysIKA {

template <typename Scalar, int Dim>
class Transform
{
public:
    Transform() {}
    ~Transform() {}

private:
    void compileTimeCheck()
    {
        //Transform<Scalar,Dim> is only defined for 2D&&3D with element type of floating-point types
        //compile time check
        //PHYSIKA_STATIC_ASSERT(Dim==2||Dim==3,"Transform<Scalar,Dim> are only defined for Dim==2,3");
        //PHYSIKA_STATIC_ASSERT(is_floating_point<Scalar>::value,
        //                      "Transform<Scalar,Dim> are only defined for floating-point types.");
    }
};

}  //end of namespace PhysIKA

#endif  //PHYSIKA_CORE_TRANSFORM_TRANSFORM_H_
