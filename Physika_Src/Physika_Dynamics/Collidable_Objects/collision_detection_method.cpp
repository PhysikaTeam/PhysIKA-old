/*
 * @file  collision_detection_method.cpp
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

#include "Physika_Dynamics/Collidable_Objects/collision_detection_method.h"

namespace Physika{

template<typename Scalar, int Dim>
CollisionDetectionMethod<Scalar, Dim>::CollisionDetectionMethod()
{

}

template<typename Scalar, int Dim>
CollisionDetectionMethod<Scalar, Dim>::~CollisionDetectionMethod()
{

}


template<typename Scalar, int Dim>
void CollisionDetectionMethod<Scalar, Dim>::cleanResults()
{
    collision_pairs_.cleanCollisionPairs();
    contact_points_.cleanContactPoints();
}

template<typename Scalar, int Dim>
unsigned int CollisionDetectionMethod<Scalar, Dim>::numPCS() const
{
    return collision_pairs_.numberPCS();
}

template<typename Scalar, int Dim>
unsigned int CollisionDetectionMethod<Scalar, Dim>::numCollisionPair() const
{
    return collision_pairs_.numberCollision();
}

template<typename Scalar, int Dim>
CollisionPairBase<Scalar, Dim>* CollisionDetectionMethod<Scalar, Dim>::collisionPair(unsigned int index)
{
    return collision_pairs_.collisionPair(index);
}

template<typename Scalar, int Dim>
unsigned int CollisionDetectionMethod<Scalar, Dim>::numContactPoint() const
{
    return contact_points_.numContactPoint();
}

template<typename Scalar, int Dim>
ContactPoint<Scalar, Dim>* CollisionDetectionMethod<Scalar, Dim>::contactPoint(unsigned int index)
{
    return contact_points_.contactPoint(index);
}

template class CollisionDetectionMethod<float, 2>;
template class CollisionDetectionMethod<double, 2>;
template class CollisionDetectionMethod<float, 3>;
template class CollisionDetectionMethod<double, 3>;

}