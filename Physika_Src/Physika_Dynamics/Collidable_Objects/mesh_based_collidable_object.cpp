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

#include "Physika_Dynamics/Collidable_Objects/mesh_based_collidable_object.h"

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
typename CollidableObject<Scalar, Dim>::ObjectType MeshBasedCollidableObject<Scalar, Dim>::getObjectType() const
{
	return CollidableObject<Scalar, Dim>::MESH_BASED;
}

template <typename Scalar,int Dim>
const SurfaceMesh<Scalar>* MeshBasedCollidableObject<Scalar, Dim>::getMesh() const
{
	return mesh_;
}

template <typename Scalar,int Dim>
SurfaceMesh<Scalar>* MeshBasedCollidableObject<Scalar, Dim>::getMesh()
{
	return mesh_;
}


template <typename Scalar,int Dim>
void MeshBasedCollidableObject<Scalar, Dim>::setMesh(SurfaceMesh<Scalar>* mesh)
{
	mesh_ = mesh;
}

//explicit instantitation
template class MeshBasedCollidableObject<float, 3>;
template class MeshBasedCollidableObject<double, 3>;

}
