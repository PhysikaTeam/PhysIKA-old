/*
 * @file  bvh_node_base.cpp
 * @base class of a BVH's node
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
#include "Physika_Dynamics/Collidable_Objects/collision_detection_result.h"
#include <stdio.h>

namespace Physika{

template <typename Scalar,int Dim>
BVHNodeBase<Scalar, Dim>::BVHNodeBase():
	is_leaf_(false),
	bv_type_(BoundingVolume<Scalar, Dim>::KDOP18),
	bounding_volume_(NULL),
	left_child_(NULL),
	right_child_(NULL)
{
}

template <typename Scalar,int Dim>
BVHNodeBase<Scalar, Dim>::~BVHNodeBase()
{
	clean();
}

template <typename Scalar,int Dim>
void BVHNodeBase<Scalar, Dim>::setLeftChild(BVHNodeBase<Scalar, Dim>* left_child)
{
	if(left_child_ == left_child)
		return;
	if(left_child_ != NULL)
		delete left_child_;
	left_child_ = left_child;
}

template <typename Scalar,int Dim>
const BVHNodeBase<Scalar, Dim>* const BVHNodeBase<Scalar, Dim>::getLeftChild() const
{
	return left_child_;
}

template <typename Scalar,int Dim>
void BVHNodeBase<Scalar, Dim>::setRightChild(BVHNodeBase<Scalar, Dim>* right_child)
{
	if(right_child_ == right_child)
		return;
	if(right_child_ != NULL)
		delete right_child_;
	right_child_ = right_child;
}

template <typename Scalar,int Dim>
const BVHNodeBase<Scalar, Dim>* const BVHNodeBase<Scalar, Dim>::getRightChild() const
{
	return right_child_;
}

template <typename Scalar,int Dim>
void BVHNodeBase<Scalar, Dim>::setBoundingVolume(BoundingVolume<Scalar, Dim>* bounding_volume)
{
	if(bounding_volume_ == bounding_volume)
		return;
	if(bounding_volume_ != NULL)
		delete bounding_volume_;
	bounding_volume_ = bounding_volume;
}

template <typename Scalar,int Dim>
const BoundingVolume<Scalar, Dim>* const BVHNodeBase<Scalar, Dim>::getBoundingVolume() const
{
	return bounding_volume_;
}

template <typename Scalar,int Dim>
void BVHNodeBase<Scalar, Dim>::setBVType(typename BoundingVolume<Scalar, Dim>::BVType bv_type)
{
	bv_type_ = bv_type;
}

template <typename Scalar,int Dim>
typename BoundingVolume<Scalar, Dim>::BVType BVHNodeBase<Scalar, Dim>::getBVType() const
{
	return bv_type_;
}

template <typename Scalar,int Dim>
void BVHNodeBase<Scalar, Dim>::setLeaf(const bool is_leaf)
{
	is_leaf_ = is_leaf;
}

template <typename Scalar,int Dim>
bool BVHNodeBase<Scalar, Dim>::isLeaf() const
{
	return is_leaf_;
}

template <typename Scalar,int Dim>
bool BVHNodeBase<Scalar, Dim>::isSceneNode() const
{
	return false;
}

template <typename Scalar,int Dim>
bool BVHNodeBase<Scalar, Dim>::isObjectNode() const
{
	return false;
}

template <typename Scalar,int Dim>
void BVHNodeBase<Scalar, Dim>::resize()
{
}

template <typename Scalar,int Dim>
void BVHNodeBase<Scalar, Dim>::refit()
{
	if(is_leaf_)
	{
		this->resize();
		return;
	}
	if(left_child_ != NULL)
		left_child_->refit();
	if(right_child_ != NULL)
		right_child_->refit();
	
	bounding_volume_->obtainUnion(left_child_->getBoundingVolume(), right_child_->getBoundingVolume());
}

template <typename Scalar,int Dim>
void BVHNodeBase<Scalar, Dim>::clean()
{
	if(!is_leaf_)
	{
		if(left_child_ != NULL)
		{
			left_child_->clean();
			delete left_child_;
			left_child_ = NULL;
		}
		if(right_child_ != NULL)
		{
			right_child_->clean();
			delete right_child_;
			right_child_ = NULL;
		}
	}
	if(bounding_volume_ != NULL)
	{
		delete bounding_volume_;
		bounding_volume_ = NULL;
	}
}

template <typename Scalar,int Dim>
void BVHNodeBase<Scalar, Dim>::cleanInternalNodes()
{
	if(!is_leaf_)
	{
		if(left_child_ != NULL && !left_child_->is_leaf_)
		{
			left_child_->clean();
			delete left_child_;
			left_child_ = NULL;
		}
		if(right_child_ != NULL && !right_child_->is_leaf_)
		{
			right_child_->clean();
			delete right_child_;
			right_child_ = NULL;
		}
		if(bounding_volume_ != NULL)
		{
			delete bounding_volume_;
			bounding_volume_ = NULL;
		}
	}
}

template <typename Scalar,int Dim>
bool BVHNodeBase<Scalar, Dim>::selfCollide(CollisionDetectionResult<Scalar, Dim>& collision_result)
{
	bool isCollide = false;
	if(is_leaf_)
		return false;
	if(left_child_ != NULL && right_child_ != NULL && left_child_->collide(right_child_, collision_result))
		isCollide = true;
	if(left_child_ != NULL && left_child_->selfCollide(collision_result))
		isCollide = true;
	if(right_child_ != NULL && right_child_->selfCollide(collision_result))
		isCollide = true;
	return isCollide;
}

template <typename Scalar,int Dim>
bool BVHNodeBase<Scalar, Dim>::collide(const BVHNodeBase<Scalar, Dim>* const target, CollisionDetectionResult<Scalar, Dim>& collision_result)
{
	if(target == NULL)
		return false;
	bool isCollide = false;
	if(!bounding_volume_->isOverlap(target->getBoundingVolume()))
		return false;
	if(is_leaf_)
	{
		if(leafCollide(target, collision_result))
			isCollide = true;
	}
	else
	{
		if(left_child_ != NULL && left_child_->collide(target, collision_result))
			isCollide = true;
		if(right_child_ != NULL && right_child_->collide(target, collision_result))
			isCollide = true;
	}
	return isCollide;
}

template <typename Scalar,int Dim>
bool BVHNodeBase<Scalar, Dim>::leafCollide(const BVHNodeBase<Scalar, Dim>* const target, CollisionDetectionResult<Scalar, Dim>& collision_result)
{
	if(target == NULL)
		return false;
	bool isCollide = false;
	if(!target->isLeaf())
	{
		if(target->getLeftChild() != NULL && bounding_volume_->isOverlap(target->getLeftChild()->getBoundingVolume()))
		{
			if(leafCollide(target->getLeftChild(), collision_result))
			{
				isCollide = true;
			}
		}
		if(target->getRightChild() != NULL && bounding_volume_->isOverlap(target->getRightChild()->getBoundingVolume()))
		{
			if(leafCollide(target->getRightChild(), collision_result))
			{
				isCollide = true;
			}
		}
	}
	else
	{
		return this->elemTest(target, collision_result);
	}
	return isCollide;
}

template <typename Scalar,int Dim>
bool BVHNodeBase<Scalar, Dim>::elemTest(const BVHNodeBase<Scalar, Dim>* const target, CollisionDetectionResult<Scalar, Dim>& collision_result)
{
	return false;
}

//explicit instantitation
template class BVHNodeBase<float, 3>;
template class BVHNodeBase<double, 3>;

}