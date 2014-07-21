/*
 * @file  collision_detection_method_DTBVH.h
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

#ifndef PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_COLLISION_DETECTION_METHOD_DTBVH_H_
#define PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_COLLISION_DETECTION_METHOD_DTBVH_H_

#include "Physika_Dynamics/Collidable_Objects/collision_detection_method.h"
#include "Physika_Geometry/Bounding_Volume/bvh_base.h"
#include "Physika_Geometry/Bounding_Volume/scene_bvh.h"
#include "Physika_Geometry/Bounding_Volume/object_bvh.h"

namespace Physika{

template <typename Scalar,int Dim>
class CollisionDetectionMethodDTBVH : public CollisionDetectionMethod<Scalar, Dim>
{
public:
    CollisionDetectionMethodDTBVH();
    ~CollisionDetectionMethodDTBVH();
    void update();
    void addCollidableObject(CollidableObject<Scalar, Dim>* object);
    bool collisionDetection();
protected:
    SceneBVH<Scalar, Dim> scene_bvh_;
};

}

#endif //PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_COLLISION_DETECTION_METHOD_DTBVH_H_