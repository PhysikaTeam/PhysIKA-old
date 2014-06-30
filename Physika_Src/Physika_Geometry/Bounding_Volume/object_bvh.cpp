/*
 * @file  object_bvh.cpp
 * @bounding volume hierarchy (BVH) of a collidable object, second level of DT-BVH [Tang et al. 2009]
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

#include "Physika_Geometry/Bounding_Volume/object_bvh.h"
#include "Physika_Geometry/Bounding_Volume/object_bvh_node.h"
#include "Physika_Geometry/Bounding_Volume/bvh_base.h"
#include "Physika_Geometry/Bounding_Volume/bvh_node_base.h"
#include "Physika_Dynamics/Collidable_Objects/mesh_based_collidable_object.h"
#include "Physika_Geometry/Bounding_Volume/bounding_volume_kdop18.h"
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

template <typename Scalar,int Dim>
ObjectBVH<Scalar, Dim>::ObjectBVH():
	collidable_object_(NULL)
{
}

template <typename Scalar,int Dim>
ObjectBVH<Scalar, Dim>::~ObjectBVH()
{
	if(collidable_object_ != NULL)
		delete collidable_object_;
}

template <typename Scalar,int Dim>
const CollidableObject<Scalar, Dim>* const ObjectBVH<Scalar, Dim>::collidableObject() const
{
	return collidable_object_;
}

template <typename Scalar,int Dim>
void ObjectBVH<Scalar, Dim>::setCollidableObject(CollidableObject<Scalar, Dim>* collidable_object)
{
	if(!this->isEmpty())
		BVHBase<Scalar, Dim>::clean();
	collidable_object_ = collidable_object;
	if(collidable_object == NULL)
		return;
	if(collidable_object_->objectType() == CollidableObject<Scalar, Dim>::MESH_BASED)
		buildFromMeshObject((MeshBasedCollidableObject<Scalar, Dim>*)collidable_object_);
}

template <typename Scalar,int Dim>
void ObjectBVH<Scalar, Dim>::buildFromMeshObject(MeshBasedCollidableObject<Scalar, Dim>* collidable_object)
{
	if(!this->isEmpty())
		BVHBase<Scalar, Dim>::clean();
	if(collidable_object == NULL)
		return;
	SurfaceMesh<Scalar>* mesh = collidable_object->mesh();
	if(mesh == NULL)
		return;
	ObjectBVHNode<Scalar, Dim>* node = NULL;

	unsigned int group_num = mesh->numGroups();
    for(unsigned int group_idx = 0; group_idx < group_num; ++group_idx)
    {
		Group<Scalar>& group = mesh->group(group_idx);
		unsigned int face_num = group.numFaces();
        for(unsigned int face_idx = 0; face_idx < face_num; ++face_idx)
		{
			node = new ObjectBVHNode<Scalar, Dim>();
			node->setLeaf(true);
			node->setBVType(this->bv_type_);
			node->setObject(collidable_object);
			node->setFaceIndex(face_idx);
			this->addNode(node);
		}
    }
	this->root_node_ = BVHBase<Scalar, Dim>::buildFromLeafList(0, numLeaf());
}

template class ObjectBVH<float, 3>;
template class ObjectBVH<double, 3>;

}
