/*
 * @file PDM_collision_method_mesh_2d.cpp
 * @brief class of collision method(two dim) based on mesh for PDM drivers.
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

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_mesh_2d.h"

namespace Physika{

template <typename Scalar>
PDMCollisionMethodMesh<Scalar, 2>::PDMCollisionMethodMesh()
{

}

template <typename Scalar>
PDMCollisionMethodMesh<Scalar, 2>::~PDMCollisionMethodMesh()
{

}

template <typename Scalar>
void PDMCollisionMethodMesh<Scalar, 2>::locateParticleBin()
{
    // to do
}

template <typename Scalar>
void PDMCollisionMethodMesh<Scalar, 2>::collisionDectectionAndResponse()
{
    // to do
}

//explicit instantiations
template class PDMCollisionMethodMesh<float, 2>;
template class PDMCollisionMethodMesh<double,2>;

}//end of namespace Physika