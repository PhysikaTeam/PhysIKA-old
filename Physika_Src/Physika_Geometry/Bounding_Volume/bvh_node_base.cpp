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

#include "Physika_Geometry\Bounding_Volume\bvh_node_base.h"

namespace Physika{

template <typename Scalar,int Dim>
BVHNodeBase<Scalar, Dim>::BVHNodeBase()
{
}

template <typename Scalar,int Dim>
inline void BVHNodeBase<Scalar, Dim>::setLeftChild(BVHNodeBase<Scalar, Dim>* left_child)
{
	if(left_child_ == left_child)
		return;
	if(left_child_ != null)
		delete left_child_;
	left_child_ = left_child;
}

template <typename Scalar,int Dim>
inline BVHNodeBase<Scalar, Dim>* BVHNodeBase<Scalar, Dim>::getLeftChild()
{
	return left_child_;
}

template <typename Scalar,int Dim>
inline void BVHNodeBase<Scalar, Dim>::setRightChild(BVHNodeBase<Scalar, Dim>* right_child)
{
	if(right_child_ == right_child)
		return;
	if(right_child_ != null)
		delete right_child_;
	right_child_ = right_child;
}

template <typename Scalar,int Dim>
inline BVHNodeBase<Scalar, Dim>* BVHNodeBase<Scalar, Dim>::getRightChild()
{
	return right_child_;
}

template <typename Scalar,int Dim>
inline void BVHNodeBase<Scalar, Dim>::setBoundingVolume(BoundingVolume<Scalar, Dim>* bounding_volume)
{
	if(bounding_volume_ == bounding_volume)
		return;
	if(bounding_volume_ != null)
		delete bounding_volume_;
	bounding_volume_ = bounding_volume;
}

template <typename Scalar,int Dim>
inline BoundingVolume<Scalar, Dim>* BVHNodeBase<Scalar, Dim>::getBoundingVolume() const
{
	return bounding_volume_;
}

template <typename Scalar,int Dim>
inline bool BVHNodeBase<Scalar, Dim>::isLeaf() const
{
	return is_leaf_;
}

//template <typename Scalar,int Dim>
//inline void BVHNodeBase<Scalar, Dim>::build()
//{
//}

template <typename Scalar,int Dim>
inline void BVHNodeBase<Scalar, Dim>::refit()
{
	if(is_leaf_)
	{
		this->resize();
		return;
	}
	if(left_child_ != null)
		left_child_->refit();
	if(right_child_ != null)
		right_child_->refit();
	
	bounding_volume_->obtainUnion(*left_child_->getBoundingVolume(), *right_child_->getBoundingVolume());
}

template <typename Scalar,int Dim>
inline void BVHNodeBase<Scalar, Dim>::clean()
{
	if(!is_leaf_)
	{
		if(left_child != null)
		{
			left_child_->clean();
			delete left_child_;
		}
		if(right_child_ != null)
		{
			right_child_->clean();
			delete right_child_;
		}
	}
	delete bounding_volume_;
}

template <typename Scalar,int Dim>
inline bool BVHNodeBase<Scalar, Dim>::selfCollide()
{
}

template <typename Scalar,int Dim>
inline bool BVHNodeBase<Scalar, Dim>::collide(const BVHNodeBase<Scalar, Dim>* const target)
{
}

template <typename Scalar,int Dim>
inline bool BVHNodeBase<Scalar, Dim>::leafCollide(const BVHNodeBase<Scalar, Dim>* const target)
{
}

}