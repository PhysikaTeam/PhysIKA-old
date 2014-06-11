/*
 * @file  scene_bvh_node.h
 * @node of the scene's BVH
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

#ifndef PHYSIKA_GEOMETRY_BOUNDING_VOLUME_SCENE_BVH_NODE_H_
#define PHYSIKA_GEOMETRY_BOUNDING_VOLUME_SCENE_BVH_NODE_H_

namespace Physika{

template <typename Scalar,int Dim> class Vector;
template <typename Scalar,int Dim> class BVHNodeBase;
template <typename Scalar,int Dim> class ObjectBVH;

template <typename Scalar,int Dim>
class SceneBVHNode : public BVHNodeBase<Scalar, Dim>
{
public:
	//constructors && deconstructors
	SceneBVHNode();
	~SceneBVHNode();

	//get & set

	//structure maintain
	void resize();
	void buildFromObjectBVH();
	
protected:
	ObjectBVH<Scalar, Dim>* object_bvh_;
};

}  //end of namespace Physika

#endif  //PHYSIKA_GEOMETRY_BOUNDING_VOLUME_SCENE_BVH_NODE_H_