/*
 * @file  collision_detection_method_DTBVH.cpp
 * @collision detection using DVBVH in 
 * "ICCD: Interactive Continuous Collision Detection between Deformable Models using Connectivity-Based Culling"
 * Tang et al. 2007
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

#include "Physika_Dynamics/Collidable_Objects/collision_detection_method_DTBVH.h"

namespace Physika{

template<typename Scalar, int Dim>
CollisionDetectionMethodDTBVH<Scalar, Dim>::CollisionDetectionMethodDTBVH()
{

}

template<typename Scalar, int Dim>
CollisionDetectionMethodDTBVH<Scalar, Dim>::~CollisionDetectionMethodDTBVH()
{

}

template<typename Scalar, int Dim>
void CollisionDetectionMethodDTBVH<Scalar, Dim>::update()
{
    scene_bvh_.updateSceneBVH();
}

template<typename Scalar, int Dim>
void CollisionDetectionMethodDTBVH<Scalar, Dim>::addCollidableObject(CollidableObject<Scalar, Dim>* object)
{
    ObjectBVH<Scalar, Dim>* object_bvh = new ObjectBVH<Scalar, Dim>();
    object_bvh->setCollidableObject(object);
    scene_bvh_.addObjectBVH(object_bvh);
}

template<typename Scalar, int Dim>
bool CollisionDetectionMethodDTBVH<Scalar, Dim>::collisionDetection()
{
    bool is_collide = scene_bvh_.selfCollide(this->collision_pairs_);
    (this->contact_points_).setCollisionResult(this->collision_pairs_);
    return is_collide;

}

template class CollisionDetectionMethodDTBVH<float, 2>;
template class CollisionDetectionMethodDTBVH<double, 2>;
template class CollisionDetectionMethodDTBVH<float, 3>;
template class CollisionDetectionMethodDTBVH<double, 3>;

}