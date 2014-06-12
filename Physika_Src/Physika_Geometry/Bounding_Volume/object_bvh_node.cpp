/*
 * @file  object_bvh_node.cpp
 * @node of a collidable object's BVH
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

#include "Physika_Geometry/Bounding_Volume/bvh_node_base.h"
#include "Physika_Geometry/Bounding_Volume/object_bvh_node.h"
#include "Physika_Geometry/Bounding_Volume/bounding_volume_kdop18.h"
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

template <typename Scalar,int Dim>
ObjectBVHNode<Scalar, Dim>::ObjectBVHNode():
	face_(NULL)
{
}

template <typename Scalar,int Dim>
ObjectBVHNode<Scalar, Dim>::~ObjectBVHNode()
{
}

template <typename Scalar,int Dim>
typename CollidableObject<Scalar, Dim>::ObjectType ObjectBVHNode<Scalar, Dim>::getObjectType() const
{
	return object_type_;
}

template <typename Scalar,int Dim>
void ObjectBVHNode<Scalar, Dim>::setFace(Face<Scalar>* face)
{
	face_ = face;
	object_type_ = CollidableObject<Scalar, Dim>::MESH_BASED;
	buildFromMesh();
}

template <typename Scalar,int Dim>
const Face<Scalar>* ObjectBVHNode<Scalar, Dim>::getFace() const
{
	return face_;
}

template <typename Scalar,int Dim>
void ObjectBVHNode<Scalar, Dim>::resize()
{
	if(object_type_ == CollidableObject<Scalar, Dim>::MESH_BASED)
		buildFromMesh();
}

template <typename Scalar,int Dim>
bool ObjectBVHNode<Scalar, Dim>::elemTest(ObjectBVHNode<Scalar, Dim>* target)
{
	//TO DO
	return false;
}

template <typename Scalar,int Dim>
void ObjectBVHNode<Scalar, Dim>::buildFromMesh()
{
	this->is_leaf_ = true;
	if(face_ == NULL)
		return;
	if(this->bounding_volume_ == NULL)
	{
		switch(this->bv_type_)
		{
			case BoundingVolume<Scalar, Dim>::KDOP18: this->bounding_volume_ = new BoundingVolumeKDOP18<Scalar, Dim>();
			default: this->bounding_volume_ = new BoundingVolumeKDOP18<Scalar, Dim>();
		}
	}
	this->bounding_volume_->setEmpty();
}

template class ObjectBVHNode<float, 3>;
template class ObjectBVHNode<double, 3>;

}