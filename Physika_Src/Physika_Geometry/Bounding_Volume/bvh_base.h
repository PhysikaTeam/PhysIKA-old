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

#include<vector>
#include "Physika_Geometry/Bounding_Volume/bounding_volume.h"

namespace Physika{

template <typename Scalar,int Dim> class Vector;
template <typename Scalar,int Dim> class BVHNodeBase;
template <typename Scalar,int Dim> class CollisionPairManager;

template <typename Scalar,int Dim>
class BVHBase
{
public:
	//constructors && deconstructors
	BVHBase();
	virtual ~BVHBase();

	//get & set
	void setRootNode(BVHNodeBase<Scalar, Dim>* root_node);
	const BVHNodeBase<Scalar, Dim>* const rootNode() const;
	void setBVType(typename BoundingVolumeInternal::BVType bv_type);
	typename BoundingVolumeInternal::BVType BVType() const;
	const BoundingVolume<Scalar, Dim>* boundingVolume() const;
	unsigned int numLeaf() const;
	bool isEmpty() const;

	//add & delete
	void addNode(BVHNodeBase<Scalar, Dim>* node);
	void deleteNode(unsigned int node_index);
	void deleteNode(BVHNodeBase<Scalar, Dim>* node);
	BVHNodeBase<Scalar, Dim>* findNode(unsigned int node_index);

	//structure maintain

	//Refit the tree from bottom to top
	//It only updates BVs and doesn't change the structure of tree
	void refit();

	//Rebuild the tree from top to bottom
	//It modifies BVs as well as the structure of tree
	//*****WARNING! This function is designed for rebuilding an existing BVH from leaf_node_list_*****
	//*****To build a BVH from an object or a scene, use the function defined in child classes*****
	void rebuild();

	//Delete the whole tree, including the root itself
	void clean();

	//Delete internal nodes including the root. Leaf nodes are remained and can be traced in leaf_node_list_
	//This is useful when rebuilding the BVH
	void cleanInternalNodes();

	//collision detection
	bool selfCollide(CollisionPairManager<Scalar, Dim>& collision_result);
	bool collide(const BVHBase<Scalar, Dim>* const target, CollisionPairManager<Scalar, Dim>& collision_result);
	
protected:
	BVHNodeBase<Scalar, Dim>* root_node_;
	typename BoundingVolumeInternal::BVType bv_type_;

	//internal function

	//Build BVH from the nodes in leaf_node_list_ with indexes in [StartPosition, EndPosition)
	//Return the root of this sub-tree
	BVHNodeBase<Scalar, Dim>* buildFromLeafList(const int start_position, const int end_position);

	//Called after deleting a leaf node. Reset the indexes of all leaf nodes according to their index in the ordered leaf list.
	void resetIndex();

private:
	//This two list is private because they need to be synchronized
	//Add and delete should only be done through function addNode and deleteNode

	//Leaf nodes of BVH. Notice that this list is disordered
	std::vector<BVHNodeBase<Scalar, Dim>*> leaf_node_list_;

	//Ordered leaf nodes of BVH. Notice that this list is ordered
	//This list is only used when searching a node by its index
	std::vector<BVHNodeBase<Scalar, Dim>*> ordered_leaf_node_list_;

};

//class to partite a BV according to its longest axis
template <typename Scalar,int Dim>
class BVAxisPartition
{
public:
	BVAxisPartition(BoundingVolume<Scalar, Dim>* bounding_volume);
	bool isLeftHandSide(const Vector<Scalar, Dim>& point) const;
protected:
	int longest_axis_index_;
	Scalar axis_mid_point_;
};

}  //end of namespace Physika

#endif  //PHYSIKA_GEOMETRY_BOUNDING_VOLUME_BVH_BASE_H_