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

#include "Physika_Geometry/Bounding_Volume/bounding_volume.h"

#ifndef PHYSIKA_GEOMETRY_BOUNDING_VOLUME_BVH_NODE_BASE_H_
#define PHYSIKA_GEOMETRY_BOUNDING_VOLUME_BVH_NODE_BASE_H_

namespace Physika{

template <typename Scalar,int Dim> class Vector;
template <typename Scalar,int Dim> class BoundingVolume;
template <typename Scalar,int Dim> class CollisionDetectionResult;

template <typename Scalar,int Dim>
class BVHNodeBase
{
public:
	//constructors && deconstructors
	BVHNodeBase();
	virtual ~BVHNodeBase();

	//get & set
	void setLeftChild(BVHNodeBase<Scalar, Dim>* left_child);
	const BVHNodeBase<Scalar, Dim>* const leftChild() const;
	void setRightChild(BVHNodeBase<Scalar, Dim>* right_child);
	const BVHNodeBase<Scalar, Dim>* const rightChild() const;
	void setBoundingVolume(BoundingVolume<Scalar, Dim>* bounding_volume);
	const BoundingVolume<Scalar, Dim>* const boundingVolume() const;
	void setBVType(typename BoundingVolume<Scalar, Dim>::BVType bv_type);
	typename BoundingVolume<Scalar, Dim>::BVType BVType() const;
	void setLeaf(const bool is_leaf);
	bool isLeaf() const;
	virtual bool isSceneNode() const;
	virtual bool isObjectNode() const;

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

	//Delete internal nodes.
	void cleanInternalNodes();

	//collision detection
	bool selfCollide(CollisionDetectionResult<Scalar, Dim>& collision_result);
	bool collide(const BVHNodeBase<Scalar, Dim>* const target, CollisionDetectionResult<Scalar, Dim>& collision_result);
	bool leafCollide(const BVHNodeBase<Scalar, Dim>* const target, CollisionDetectionResult<Scalar, Dim>& collision_result);
	virtual bool elemTest(const BVHNodeBase<Scalar, Dim>* const target, CollisionDetectionResult<Scalar, Dim>& collision_result);
	
protected:
	bool is_leaf_;
	typename BoundingVolume<Scalar, Dim>::BVType bv_type_;
	BoundingVolume<Scalar, Dim>* bounding_volume_;
	BVHNodeBase<Scalar, Dim>* left_child_;
	BVHNodeBase<Scalar, Dim>* right_child_;
};

}  //end of namespace Physika

#endif  //PHYSIKA_GEOMETRY_BOUNDING_VOLUME_BVH_NODE_BASE_H_