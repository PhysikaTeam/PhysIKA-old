/*
 * @file  collision_detection_method_CCD.h
 * @continuous collision detection using ICCD in 
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

#ifndef PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_COLLISION_DETECTION_METHOD_CCD_H_
#define PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_COLLISION_DETECTION_METHOD_CCD_H_

#include "Physika_Dynamics/Collidable_Objects/collision_detection_method.h"
#include "Physika_Geometry/Bounding_Volume/bvh_base.h"
#include "Physika_Geometry/Bounding_Volume/scene_bvh.h"
#include "Physika_Geometry/Bounding_Volume/object_bvh.h"

namespace Physika{

template <typename Scalar,int Dim>
class CollisionDetectionMethodCCD : public CollisionDetectionMethod<Scalar, Dim>
{
public:
    CollisionDetectionMethodCCD();
    ~CollisionDetectionMethodCCD();
    void update();
    void addCollidableObject(CollidableObject<Scalar, Dim>* object);
    bool collisionDetection();
protected:

    void continuousSampling();

    SceneBVH<Scalar, Dim> scene_bvh_;
    std::vector<CollidableObject<Scalar, Dim>* > previous_objects_;//record the status of bodies in the previous step. This will be used in continuous collision detection.
};

}

#endif //PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_COLLISION_DETECTION_METHOD_CCD_H_