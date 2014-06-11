/*
 * @file  bvh_node_base.h
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

#ifndef PHYSIKA_GEOMETRY_BOUNDING_VOLUME_BVH_NODE_BASE_H_
#define PHYSIKA_GEOMETRY_BOUNDING_VOLUME_BVH_NODE_BASE_H_

namespace Physika{

template <typename Scalar,int Dim> class Vector;
template <typename Scalar,int Dim> class BoundingVolume;

template <typename Scalar,int Dim>
class BVHNodeBase
{
public:
	//constructors && deconstructors
	BVHNodeBase();
	virtual ~BVHNodeBase();

	//get & set
	void setLeftChild(BVHNodeBase<Scalar, Dim>* left_child);
	const BVHNodeBase<Scalar, Dim>* const getLeftChild() const;
	void setRightChild(BVHNodeBase<Scalar, Dim>* right_child);
	const BVHNodeBase<Scalar, Dim>* const getRightChild() const;
	void setBoundingVolume(BoundingVolume<Scalar, Dim>* bounding_volume);
	const BoundingVolume<Scalar, Dim>* const getBoundingVolume() const;
	bool isLeaf() const;

	//structure maintain
	
	//Resize the BV according to the content or children's BVs
	virtual void resize();

	//Refit the sub-tree from bottom to top
	//It only updates BVs and doesn't change the structure of tree
	void refit();

	//Delete its sub-tree and content
	//It doesn't delete the node itself
	//*****WARNING! Contents defined in the child class are not deleted by this function!*****
	//*****If you want to delete contents defined in the child, add a new cleaning function in the child class!*****
	void clean();

	//collision detection
	bool selfCollide();
	bool collide(const BVHNodeBase<Scalar, Dim>* const target);
	bool leafCollide(const BVHNodeBase<Scalar, Dim>* const target);

	
protected:
	bool is_leaf_;
	BoundingVolume<Scalar, Dim>* bounding_volume_;
	BVHNodeBase<Scalar, Dim>* left_child_;
	BVHNodeBase<Scalar, Dim>* right_child_;
};

}  //end of namespace Physika

#endif  //PHYSIKA_GEOMETRY_BOUNDING_VOLUME_BVH_NODE_BASE_H_