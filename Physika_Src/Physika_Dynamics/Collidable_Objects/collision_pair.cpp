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
CollisionPairMeshToMesh<Scalar, Dim>::CollisionPairMeshToMesh(unsigned int object_lhs_index, unsigned int object_rhs_index, 
															MeshBasedCollidableObject<Scalar, Dim>* object_lhs, MeshBasedCollidableObject<Scalar, Dim>* object_rhs, 
															unsigned int face_lhs_index, unsigned int face_rhs_index):
	object_lhs_index_(object_lhs_index),
	object_rhs_index_(object_rhs_index),
	object_lhs_(object_lhs),
	object_rhs_(object_rhs),
	face_lhs_index_(face_lhs_index),
	face_rhs_index_(face_rhs_index)
{
	face_lhs_ = object_lhs->mesh()->facePtr(face_lhs_index);
	face_rhs_ = object_rhs->mesh()->facePtr(face_rhs_index);
}

template <typename Scalar,int Dim>
CollisionPairMeshToMesh<Scalar, Dim>::~CollisionPairMeshToMesh()
{
}

template <typename Scalar,int Dim>
typename CollidableObjectInternal::ObjectType CollisionPairMeshToMesh<Scalar, Dim>::objectTypeLhs() const
{
    return CollidableObjectInternal::MESH_BASED;
}

template <typename Scalar,int Dim>
typename CollidableObjectInternal::ObjectType CollisionPairMeshToMesh<Scalar, Dim>::objectTypeRhs() const
{
    return CollidableObjectInternal::MESH_BASED;
}

template <typename Scalar,int Dim>
const CollidableObject<Scalar, Dim>* CollisionPairMeshToMesh<Scalar, Dim>::objectLhs() const
{
	return object_lhs_;
}

template <typename Scalar,int Dim>
CollidableObject<Scalar, Dim>* CollisionPairMeshToMesh<Scalar, Dim>::objectLhs()
{
	return object_lhs_;
}

template <typename Scalar,int Dim>
const CollidableObject<Scalar, Dim>* CollisionPairMeshToMesh<Scalar, Dim>::objectRhs() const
{
	return object_rhs_;
}

template <typename Scalar,int Dim>
CollidableObject<Scalar, Dim>* CollisionPairMeshToMesh<Scalar, Dim>::objectRhs()
{
	return object_rhs_;
}

template <typename Scalar,int Dim>
const MeshBasedCollidableObject<Scalar, Dim>* CollisionPairMeshToMesh<Scalar, Dim>::meshObjectLhs() const
{
	return object_lhs_;
}

template <typename Scalar,int Dim>
MeshBasedCollidableObject<Scalar, Dim>* CollisionPairMeshToMesh<Scalar, Dim>::meshObjectLhs()
{
	return object_lhs_;
}

template <typename Scalar,int Dim>
const MeshBasedCollidableObject<Scalar, Dim>* CollisionPairMeshToMesh<Scalar, Dim>::meshObjectRhs() const
{
	return object_rhs_;
}

template <typename Scalar,int Dim>
MeshBasedCollidableObject<Scalar, Dim>* CollisionPairMeshToMesh<Scalar, Dim>::meshObjectRhs()
{
	return object_rhs_;
}

template <typename Scalar,int Dim>
const Face<Scalar>* CollisionPairMeshToMesh<Scalar, Dim>::faceLhsPtr() const
{
	return face_lhs_;
}

template <typename Scalar,int Dim>
Face<Scalar>* CollisionPairMeshToMesh<Scalar, Dim>::faceLhsPtr()
{
	return face_lhs_;
}

template <typename Scalar,int Dim>
const Face<Scalar>* CollisionPairMeshToMesh<Scalar, Dim>::faceRhsPtr() const
{
	return face_rhs_;
}

template <typename Scalar,int Dim>
Face<Scalar>* CollisionPairMeshToMesh<Scalar, Dim>::faceRhsPtr()
{
	return face_rhs_;
}

template <typename Scalar,int Dim>
unsigned int CollisionPairMeshToMesh<Scalar, Dim>::faceLhsIdx() const
{
	return face_lhs_index_;
}

template <typename Scalar,int Dim>
unsigned int CollisionPairMeshToMesh<Scalar, Dim>::faceRhsIdx() const
{
	return face_rhs_index_;
}

template <typename Scalar,int Dim>
unsigned int CollisionPairMeshToMesh<Scalar, Dim>::objectLhsIdx() const
{
	return object_lhs_index_;
}

template <typename Scalar,int Dim>
unsigned int CollisionPairMeshToMesh<Scalar, Dim>::objectRhsIdx() const
{
	return object_rhs_index_;
}

template class CollisionPairBase<float, 3>;
template class CollisionPairBase<double, 3>;
template class CollisionPairMeshToMesh<float, 3>;
template class CollisionPairMeshToMesh<double, 3>;

}