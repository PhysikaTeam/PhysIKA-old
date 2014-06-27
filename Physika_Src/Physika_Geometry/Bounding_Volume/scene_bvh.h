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
template <typename Scalar,int Dim> class BVHBase;
template <typename Scalar,int Dim> class SceneBVHNode;
template <typename Scalar,int Dim> class ObjectBVH;

template <typename Scalar,int Dim>
class SceneBVH : public BVHBase<Scalar, Dim>
{
public:
	//constructors && deconstructors
	SceneBVH();
	~SceneBVH();

	//get & set

	//structure maintain
	void addObjectBVH(ObjectBVH<Scalar, Dim>* object_bvh, bool isRebuild = true);
	void refitLeafNodes();
	//Update the scene BVH. First refit leaf nodes, then rebuild the scene BVH
	void updateSceneBVH();
	
protected:

};

}  //end of namespace Physika

#endif  //PHYSIKA_GEOMETRY_BOUNDING_VOLUME_SCENE_BVH_H_