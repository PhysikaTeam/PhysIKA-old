/*
 * @file  collision_detection_method.h
 * @Base class of collision detection methods
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

#ifndef PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_COLLISION_DETECTION_METHOD_H_
#define PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_COLLISION_DETECTION_METHOD_H_

#include "Physika_Dynamics/Collidable_Objects/collision_pair.h"
#include "Physika_Dynamics/Collidable_Objects/collision_pair_manager.h"
#include "Physika_Dynamics/Collidable_Objects/contact_point.h"
#include "Physika_Dynamics/Collidable_Objects/contact_point_manager.h"

namespace Physika{

template <typename Scalar, int Dim> class CollidableObject;

template <typename Scalar,int Dim>
class CollisionDetectionMethod
{
public:
    //constructor
    CollisionDetectionMethod();
    virtual ~CollisionDetectionMethod();

    //dynamic function used in a driver
    virtual void update() = 0;
    virtual void addCollidableObject(CollidableObject<Scalar, Dim>* object) = 0;
    virtual bool collisionDetection() = 0;
    virtual void cleanResults();

    //getter
    unsigned int numPCS() const;
    unsigned int numCollisionPair() const;
    CollisionPairBase<Scalar, Dim>* collisionPair(unsigned int index);
    unsigned int numContactPoint() const;
    ContactPoint<Scalar, Dim>* contactPoint(unsigned int index);

protected:
    CollisionPairManager<Scalar, Dim> collision_pairs_;
    ContactPointManager<Scalar, Dim> contact_points_;
};

}

#endif //PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_COLLISION_DETECTION_METHOD_H_