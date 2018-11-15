/*
 * @file PDM_collision_method_space_hash_2d.cpp
 * @brief class of collision method(two dim) for PDM drivers.
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <iostream>
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_space_hash_2d.h"

namespace Physika{

template <typename Scalar>
PDMCollisionMethodSpaceHash<Scalar, 2>::PDMCollisionMethodSpaceHash()
{

}

template <typename Scalar>
PDMCollisionMethodSpaceHash<Scalar, 2>::~PDMCollisionMethodSpaceHash()
{

}

template <typename Scalar>
void PDMCollisionMethodSpaceHash<Scalar, 2>::collisionMethod()
{
    std::cerr<<"PDMCollisionMethodSpaceHash<Scalar, 2> currently is not implemented!\n";
    std::exit(EXIT_FAILURE);
}

}//end of namespace Physika