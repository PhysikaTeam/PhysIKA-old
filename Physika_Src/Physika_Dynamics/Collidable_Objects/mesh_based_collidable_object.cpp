/*
 * @file  mesh_based_collidable_object.cpp
 * @collidable object based on the mesh of object
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

#include "Physika_Dynamics\Collidable_Objects\mesh_based_collidable_object.h"

namespace Physika{

template <typename Scalar,int Dim>
MeshBasedCollidableObject<Scalar, Dim>::MeshBasedCollidableObject()
{
}

template <typename Scalar,int Dim>
MeshBasedCollidableObject<Scalar, Dim>::~MeshBasedCollidableObject()
{
}

template <typename Scalar,int Dim>
inline typename CollidableObject<Scalar, Dim>::ObjectType MeshBasedCollidableObject<Scalar, Dim>::getObjectType() const
{
	return MESH_BASED;
}

}