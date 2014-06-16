/*
 * @file  collision_pair.cpp
 * @pairs of colliding elementaries
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

#include "Physika_Dynamics/Collidable_Objects/collision_pair.h"
#include "Physika_Dynamics/Collidable_Objects/mesh_based_collidable_object.h"

namespace Physika{

template <typename Scalar,int Dim>
CollisionPairBase<Scalar, Dim>::CollisionPairBase()
{
}

template <typename Scalar,int Dim>
CollisionPairBase<Scalar, Dim>::~CollisionPairBase()
{
}

template <typename Scalar,int Dim>
CollisionPairMesh2Mesh<Scalar, Dim>::CollisionPairMesh2Mesh(MeshBasedCollidableObject<Scalar, Dim>* object_lhs, MeshBasedCollidableObject<Scalar, Dim>* object_rhs, Face<Scalar>* face_lhs, Face<Scalar>* face_rhs):
	object_lhs_(object_lhs),
	object_rhs_(object_rhs),
	face_lhs_(face_lhs),
	face_rhs_(face_rhs)
{
}

template <typename Scalar,int Dim>
CollisionPairMesh2Mesh<Scalar, Dim>::~CollisionPairMesh2Mesh()
{
}

template <typename Scalar,int Dim>
const CollidableObject<Scalar, Dim>* CollisionPairMesh2Mesh<Scalar, Dim>::ObjectLhs() const
{
	return object_lhs_;
}

template <typename Scalar,int Dim>
CollidableObject<Scalar, Dim>* CollisionPairMesh2Mesh<Scalar, Dim>::ObjectLhs()
{
	return object_lhs_;
}

template <typename Scalar,int Dim>
const CollidableObject<Scalar, Dim>* CollisionPairMesh2Mesh<Scalar, Dim>::ObjectRhs() const
{
	return object_rhs_;
}

template <typename Scalar,int Dim>
CollidableObject<Scalar, Dim>* CollisionPairMesh2Mesh<Scalar, Dim>::ObjectRhs()
{
	return object_rhs_;
}

template <typename Scalar,int Dim>
const MeshBasedCollidableObject<Scalar, Dim>* CollisionPairMesh2Mesh<Scalar, Dim>::MeshObjectLhs() const
{
	return object_lhs_;
}

template <typename Scalar,int Dim>
MeshBasedCollidableObject<Scalar, Dim>* CollisionPairMesh2Mesh<Scalar, Dim>::MeshObjectLhs()
{
	return object_lhs_;
}

template <typename Scalar,int Dim>
const MeshBasedCollidableObject<Scalar, Dim>* CollisionPairMesh2Mesh<Scalar, Dim>::MeshObjectRhs() const
{
	return object_rhs_;
}

template <typename Scalar,int Dim>
MeshBasedCollidableObject<Scalar, Dim>* CollisionPairMesh2Mesh<Scalar, Dim>::MeshObjectRhs()
{
	return object_rhs_;
}

template <typename Scalar,int Dim>
const Face<Scalar>* CollisionPairMesh2Mesh<Scalar, Dim>::FaceLhs() const
{
	return face_lhs_;
}

template <typename Scalar,int Dim>
Face<Scalar>* CollisionPairMesh2Mesh<Scalar, Dim>::FaceLhs()
{
	return face_lhs_;
}

template <typename Scalar,int Dim>
const Face<Scalar>* CollisionPairMesh2Mesh<Scalar, Dim>::FaceRhs() const
{
	return face_rhs_;
}

template <typename Scalar,int Dim>
Face<Scalar>* CollisionPairMesh2Mesh<Scalar, Dim>::FaceRhs()
{
	return face_rhs_;
}

template class CollisionPairBase<float, 3>;
template class CollisionPairBase<double, 3>;
template class CollisionPairMesh2Mesh<float, 3>;
template class CollisionPairMesh2Mesh<double, 3>;

}