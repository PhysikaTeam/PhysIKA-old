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
template <typename Scalar,int Dim> class BVHNodeBase;

template <typename Scalar,int Dim>
class ObjectBVHNode : public BVHNodeBase<Scalar, Dim>
{
public:
	//constructors && deconstructors
	ObjectBVHNode();
	~ObjectBVHNode();

	//structure maintain
	void resize();
	void buildFromShape();

	bool elemTest(ObjectBVHNode<Scalar, Dim>* target);
	
protected:

};

}  //end of namespace Physika

#endif  //PHYSIKA_GEOMETRY_BOUNDING_VOLUME_OBJECT_BVH_NODE_H_