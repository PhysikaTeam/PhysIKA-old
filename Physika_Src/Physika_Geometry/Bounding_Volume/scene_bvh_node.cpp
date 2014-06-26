/*
 * @file  scene_bvh_node.h
 * @node of the scene's BVH
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

#include "Physika_Geometry/Bounding_Volume/scene_bvh.h"
#include "Physika_Geometry/Bounding_Volume/scene_bvh_node.h"
#include "Physika_Geometry/Bounding_Volume/object_bvh.h"
#include "Physika_Geometry/Bounding_Volume/object_bvh_node.h"
#include "Physika_Geometry/Bounding_Volume/bvh_base.h"
#include "Physika_Geometry/Bounding_Volume/bvh_node_base.h"
#include "Physika_Dynamics/Collidable_Objects/mesh_based_collidable_object.h"
#include "Physika_Geometry/Bounding_Volume/bounding_volume_kdop18.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Dynamics/Collidable_Objects/collision_detection_result.h"

namespace Physika{

template <typename Scalar,int Dim>
SceneBVHNode<Scalar, Dim>::SceneBVHNode():
	object_bvh_(NULL)
{

}

template <typename Scalar,int Dim>
SceneBVHNode<Scalar, Dim>::~SceneBVHNode()
{

}

template <typename Scalar,int Dim>
bool SceneBVHNode<Scalar, Dim>::isObjectNode() const
{
	return false;
}

template <typename Scalar,int Dim>
bool SceneBVHNode<Scalar, Dim>::isSceneNode() const
{
	return true;
}

template <typename Scalar,int Dim>
const ObjectBVH<Scalar, Dim>* SceneBVHNode<Scalar, Dim>::objectBVH() const
{
	return object_bvh_;
}

template <typename Scalar,int Dim>
ObjectBVH<Scalar, Dim>* SceneBVHNode<Scalar, Dim>::objectBVH()
{
	return object_bvh_;
}

template <typename Scalar,int Dim>
void SceneBVHNode<Scalar, Dim>::setObjectBVH(ObjectBVH<Scalar, Dim>* object_bvh)
{
	object_bvh_ = object_bvh;
	this->bv_type_ = object_bvh->BVType();
	buildFromObjectBVH();
}

template <typename Scalar,int Dim>
void SceneBVHNode<Scalar, Dim>::resize()
{
	if(object_bvh_ == NULL)
		return;
	object_bvh_->refit();
	buildFromObjectBVH();
}

template <typename Scalar,int Dim>
bool SceneBVHNode<Scalar, Dim>::elemTest(const BVHNodeBase<Scalar, Dim>* const target, CollisionDetectionResult<Scalar, Dim>& collision_result)
{
	if(target == NULL || !target->isSceneNode())
		return false;
	const SceneBVHNode<Scalar, Dim>* object_target = dynamic_cast<const SceneBVHNode<Scalar, Dim>*>(target);
	if(object_target == NULL)
		return false;
	return object_bvh_->collide(object_target->objectBVH(), collision_result);
}

template <typename Scalar,int Dim>
void SceneBVHNode<Scalar, Dim>::buildFromObjectBVH()
{
	if(object_bvh_ == NULL)
		return;
	this->is_leaf_ = true;
	if(this->bounding_volume_ == NULL)
	{
		switch(this->bv_type_)
		{
		case BoundingVolume<Scalar, Dim>::KDOP18: this->bounding_volume_ = new BoundingVolumeKDOP18<Scalar, Dim>();
		default: this->bounding_volume_ = new BoundingVolumeKDOP18<Scalar, Dim>();
		}
	}
	this->bounding_volume_->setEmpty();
	this->bounding_volume_->setBoundingVolume(object_bvh_->boundingVolume());
}

template class SceneBVHNode<float, 3>;
template class SceneBVHNode<double, 3>;


}