/*
 * @file  bvh_base.cpp
 * @base class of bounding volume hierarchy (BVH)
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

#include "Physika_Geometry\Bounding_Volume\bvh_base.h"
#include "Physika_Geometry\Bounding_Volume\bvh_node_base.h"
#include <stdio.h>

namespace Physika{

template <typename Scalar,int Dim>
BVHBase<Scalar, Dim>::BVHBase():
	root_node_(NULL)
{
}

template <typename Scalar,int Dim>
BVHBase<Scalar, Dim>::~BVHBase()
{
	if(root_node_ != NULL)
		delete root_node_;
}

template <typename Scalar,int Dim>
void BVHBase<Scalar, Dim>::setRootNode(BVHNodeBase<Scalar, Dim>* root_node)
{
	if(root_node_ == root_node)
		return;
	if(root_node_ != NULL)
		delete root_node_;
	root_node_ = root_node;
}

template <typename Scalar,int Dim>
const BVHNodeBase<Scalar, Dim>* const BVHBase<Scalar, Dim>::getRootNode() const
{
	return root_node_;
}

template <typename Scalar,int Dim>
void BVHBase<Scalar, Dim>::refit()
{
	root_node_->refit();
}

template <typename Scalar,int Dim>
void BVHBase<Scalar, Dim>::build()
{
}

template <typename Scalar,int Dim>
void BVHBase<Scalar, Dim>::clean()
{
	root_node_->clean();
	delete root_node_;
}

template <typename Scalar,int Dim>
bool BVHBase<Scalar, Dim>::selfCollide()
{
	return root_node_->selfCollide();
}

template <typename Scalar,int Dim>
bool BVHBase<Scalar, Dim>::collide(const BVHBase<Scalar, Dim>* const target)
{
	return root_node_->collide(target->getRootNode());
}

//explicit instantitation
template class BVHBase<float, 3>;
template class BVHBase<double, 3>;

}