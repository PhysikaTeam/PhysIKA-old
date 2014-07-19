/*
 * @file  scene_bvh.cpp
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

#include "Physika_Geometry/Bounding_Volume/scene_bvh.h"
#include "Physika_Geometry/Bounding_Volume/scene_bvh_node.h"
#include "Physika_Geometry/Bounding_Volume/object_bvh.h"
#include "Physika_Geometry/Bounding_Volume/object_bvh_node.h"
#include "Physika_Geometry/Bounding_Volume/bvh_base.h"
#include "Physika_Geometry/Bounding_Volume/bvh_node_base.h"
#include "Physika_Dynamics/Collidable_Objects/mesh_based_collidable_object.h"
#include "Physika_Geometry/Bounding_Volume/bounding_volume_kdop18.h"
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

template <typename Scalar,int Dim>
SceneBVH<Scalar, Dim>::SceneBVH()
{

}

template <typename Scalar,int Dim>
SceneBVH<Scalar, Dim>::~SceneBVH()
{

}

template <typename Scalar,int Dim>
void SceneBVH<Scalar, Dim>::addObjectBVH(ObjectBVH<Scalar, Dim>* object_bvh, bool is_rebuild)
{
	SceneBVHNode<Scalar, Dim>* scene_node = new SceneBVHNode<Scalar, Dim>();
	scene_node->setObjectBVH(object_bvh);
	scene_node->setLeaf(true);
	scene_node->setBVType(object_bvh->BVType());
	this->addNode(scene_node);
	if(is_rebuild)
		this->rebuild();
}

template <typename Scalar,int Dim>
void SceneBVH<Scalar, Dim>::refitLeafNodes()
{
	unsigned int leaf_num = this->numLeaf();
	for(unsigned int i = 0; i < leaf_num; i++)
	{
		this->findNode(i)->resize();
	}
}

template <typename Scalar,int Dim>
void SceneBVH<Scalar, Dim>::updateSceneBVH()
{
	refitLeafNodes();
	this->rebuild();
}

template class SceneBVH<float, 2>;
template class SceneBVH<double, 2>;
template class SceneBVH<float, 3>;
template class SceneBVH<double, 3>;


}