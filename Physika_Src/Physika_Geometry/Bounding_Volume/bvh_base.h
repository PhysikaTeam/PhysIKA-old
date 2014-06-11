/*
 * @file  bvh_base.h
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

#ifndef PHYSIKA_GEOMETRY_BOUNDING_VOLUME_BVH_BASE_H_
#define PHYSIKA_GEOMETRY_BOUNDING_VOLUME_BVH_BASE_H_

namespace Physika{

template <typename Scalar,int Dim> class Vector;
template <typename Scalar,int Dim> class BVHNodeBase;

template <typename Scalar,int Dim>
class BVHBase
{
public:
	//constructors && deconstructors
	BVHBase();
	virtual ~BVHBase();

	//get & set
	void setRootNode(BVHNodeBase<Scalar, Dim>* root_node);
	const BVHNodeBase<Scalar, Dim>* const getRootNode() const;

	//structure maintain
	void refit();
	void build();
	//Delete the whole tree, including the root itself
	void clean();

	//collision detection
	bool selfCollide();
	bool collide(const BVHBase<Scalar, Dim>* const target);
	
protected:
	BVHNodeBase<Scalar, Dim>* root_node_;
};

}  //end of namespace Physika

#endif  //PHYSIKA_GEOMETRY_BOUNDING_VOLUME_BVH_BASE_H_