/*
 * @file  object_bvh_node.h
 * @node of a collidable object's BVH
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

#ifndef PHYSIKA_GEOMETRY_BOUNDING_VOLUME_OBJECT_BVH_NODE_H_
#define PHYSIKA_GEOMETRY_BOUNDING_VOLUME_OBJECT_BVH_NODE_H_

namespace Physika{

template <typename Scalar,int Dim> class Vector;
template <typename Scalar,int Dim> class BoundingVolume;

template <typename Scalar,int Dim>
class ObjectBVHNode
{
public:
	//constructors && deconstructors
	ObjectBVHNode();
	~ObjectBVHNode();

	//get & set
	inline void setLeftChild(ObjectBVHNode* left_child);
	inline ObjectBVHNode* getLeftChild();
	inline void setRightChild(ObjectBVHNode* right_child);
	inline ObjectBVHNode* getRightChild();
	inline void setBoundingVolume(BoundingVolume* bounding_volume);
	inline BoundingVolume* getBoundingVolume();

	//structure maintain
	void buildFromShape();
	void clean();

	//collision detection
	bool selfCollide();
	bool collide(ObjectBVHNode* target);
	bool leafCollide(ObjectBVHNode* target);
	bool elemTest(ObjectBVHNode* target);

	
protected:
	BoundingVolume* bounding_volume_;
	ObjectBVHNode* left_child_;
	ObjectBVHNode* right_child_;
};

}  //end of namespace Physika

#endif  //PHYSIKA_GEOMETRY_BOUNDING_VOLUME_OBJECT_BVH_NODE_H_