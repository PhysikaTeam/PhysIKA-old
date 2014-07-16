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
#include "Physika_Geometry/Bounding_Volume/bounding_volume_octagon.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"
#include "Physika_Dynamics/Collidable_Objects/mesh_based_collidable_object.h"
#include "Physika_Dynamics/Collidable_Objects/collision_detection_result.h"

namespace Physika{

template <typename Scalar,int Dim>
ObjectBVHNode<Scalar, Dim>::ObjectBVHNode():
	object_type_(CollidableObjectInternal::MESH_BASED),
	object_(NULL),
	face_index_(0),
	has_face_(false)
{
}

template <typename Scalar,int Dim>
ObjectBVHNode<Scalar, Dim>::~ObjectBVHNode()
{
}

template <typename Scalar,int Dim>
bool ObjectBVHNode<Scalar, Dim>::isSceneNode() const
{
	return false;
}

template <typename Scalar,int Dim>
bool ObjectBVHNode<Scalar, Dim>::isObjectNode() const
{
	return true;
}

template <typename Scalar,int Dim>
typename CollidableObjectInternal::ObjectType ObjectBVHNode<Scalar, Dim>::objectType() const
{
	return object_type_;
}

template <typename Scalar,int Dim>
void ObjectBVHNode<Scalar, Dim>::setObject(CollidableObject<Scalar, Dim>* object)
{
	object_ = object;
	object_type_ = object->objectType();
}

template <typename Scalar,int Dim>
const CollidableObject<Scalar, Dim>* ObjectBVHNode<Scalar, Dim>::object() const
{
	return object_;
}

template <typename Scalar,int Dim>
void ObjectBVHNode<Scalar, Dim>::setFaceIndex(unsigned int face_index)
{
	face_index_ = face_index;
	has_face_ = true;
	object_type_ = CollidableObjectInternal::MESH_BASED;
	buildFromFace();
}

template <typename Scalar,int Dim>
unsigned int ObjectBVHNode<Scalar, Dim>::getFaceIndex() const
{
	return face_index_;
}

template <typename Scalar,int Dim>
void ObjectBVHNode<Scalar, Dim>::resize()
{
	if(object_type_ == CollidableObjectInternal::MESH_BASED)
		buildFromFace();
}

template <typename Scalar,int Dim>
bool ObjectBVHNode<Scalar, Dim>::elemTest(const BVHNodeBase<Scalar, Dim>* const target, CollisionDetectionResult<Scalar, Dim>& collision_result)
{
	if(target == NULL || !target->isObjectNode())
		return false;
	if(target->BVType() != this->bv_type_)
		return false;
	const ObjectBVHNode<Scalar, Dim>* object_target = dynamic_cast<const ObjectBVHNode<Scalar, Dim>*>(target);
	if(object_target == NULL)
		return false;
	if(object_type_ == CollidableObjectInternal::MESH_BASED && object_target->objectType() == CollidableObjectInternal::MESH_BASED)
	{
		MeshBasedCollidableObject<Scalar>* mesh_object_this = dynamic_cast<MeshBasedCollidableObject<Scalar>*>(object_);
		MeshBasedCollidableObject<Scalar>* mesh_object_target = dynamic_cast<MeshBasedCollidableObject<Scalar>*>(object_target->object_);
		if(mesh_object_this == NULL || mesh_object_target == NULL)
			return false;
		if(!has_face_ || !object_target->has_face_)
			return false;
		bool is_collide = mesh_object_this->collideWithMesh(mesh_object_target, face_index_, object_target->face_index_);
		collision_result.addPCS();
		if(is_collide)
			collision_result.addCollisionPair(mesh_object_this, mesh_object_target, face_index_, object_target->face_index_);
	}
	return false;
}

template <typename Scalar, int Dim>
void ObjectBVHNode<Scalar, Dim>::buildFromFace()
{
    if(Dim == 2)
    {
        std::cerr<<"Can't build a 2D BVH from a 3D mesh!"<<std::endl;
        return;
    }
	this->is_leaf_ = true;
	if(!has_face_)
		return;
	if(object_->objectType() != CollidableObjectInternal::MESH_BASED)
		return;
	if(this->bounding_volume_ == NULL)
	{
        this->bounding_volume_ = BoundingVolumeInternal::createBoundingVolume<Scalar, Dim>(this->bv_type_);
	}
	this->bounding_volume_->setEmpty();
	const MeshBasedCollidableObject<Scalar>* const object = dynamic_cast<MeshBasedCollidableObject<Scalar>*>(object_);
	if(object == NULL)
		return;
	const Face<Scalar>& face = object->mesh()->face(face_index_);
	unsigned int point_num = face.numVertices();
	for(unsigned int i = 0; i < point_num; ++i)
	{
		this->bounding_volume_->unionWith(*dynamic_cast<Vector<Scalar, Dim>* >(&(object->vertexPosition(face.vertex(i).positionIndex()))));
	}
}

template class ObjectBVHNode<float, 2>;
template class ObjectBVHNode<double, 2>;
template class ObjectBVHNode<float, 3>;
template class ObjectBVHNode<double, 3>;

}