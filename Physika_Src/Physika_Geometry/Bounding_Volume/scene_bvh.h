/*
 * @file  scene_bvh.h
 * @bounding volume hierarchy (BVH) of the simulation scene, first level of DT-BVH [Tang et al. 2009]
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

#ifndef PHYSIKA_GEOMETRY_BOUNDING_VOLUME_SCENE_BVH_H_
#define PHYSIKA_GEOMETRY_BOUNDING_VOLUME_SCENE_BVH_H_

namespace Physika{

template <typename Scalar,int Dim> class Vector;
template <typename Scalar,int Dim> class ObjectBVH;
template <typename Scalar,int Dim> class CollidableObject;

template <typename Scalar,int Dim>
class ObjectBVH
{
public:
	//constructors && deconstructors
	ObjectBVH();
	~ObjectBVH();

	//get & set
	inline void setRootNode(ObjectBVH* root_node);
	inline ObjectBVH* getRootNode();

	//structure maintain
	void refit();
	void buildFromScene();
	void clean();

	//collision detection
	bool selfCollide();
	
protected:
	ObjectBVH* root_node_;
};

}  //end of namespace Physika

#endif  //PHYSIKA_GEOMETRY_BOUNDING_VOLUME_SCENE_BVH_H_