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

namespace Physika{

template <typename Scalar,int Dim>
BVHBase<Scalar, Dim>::BVHBase()
{
}

template <typename Scalar,int Dim>
inline void BVHBase<Scalar, Dim>::setRootNode(BVHNodeBase<Scalar, Dim>* root_node)
{
	if(root_node_ == root_node)
		return;
	if(root_node_ != null)
		delete root_node_;
	root_node_ = root_node;
}

template <typename Scalar,int Dim>
inline BVHNodeBase<Scalar, Dim>* BVHBase<Scalar, Dim>::getRootNode()
{
	return root_node_;
}

template <typename Scalar,int Dim>
inline void BVHBase<Scalar, Dim>::refit()
{
	root_node_->refit();
}

template <typename Scalar,int Dim>
inline void BVHBase<Scalar, Dim>::build()
{
}

template <typename Scalar,int Dim>
inline void BVHBase<Scalar, Dim>::clean()
{
	root_node_->clean();
	delete root_node_;
}

template <typename Scalar,int Dim>
inline bool BVHBase<Scalar, Dim>::selfCollide()
{
}

template <typename Scalar,int Dim>
inline bool BVHBase<Scalar, Dim>::collide(BVHBase<Scalar, Dim>* target)
{
}


}