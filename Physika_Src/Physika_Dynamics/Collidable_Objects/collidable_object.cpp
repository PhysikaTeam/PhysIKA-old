/*
 * @file  collidable_object.cpp
 * @brief abstract base class of all collidable objects, provide common interface
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

#include "Physika_Dynamics/Collidable_Objects/collidable_object.h"
#include "Physika_Dynamics/Collidable_Objects/collision_detection_method.h"
#include "Physika_Dynamics/Collidable_Objects/collision_detection_method_DTBVH.h"
#include "Physika_Dynamics/Collidable_Objects/collision_pair.h"
#include "Physika_Dynamics/Collidable_Objects/collision_pair_manager.h"
#include "Physika_Dynamics/Collidable_Objects/contact_point.h"
#include "Physika_Dynamics/Collidable_Objects/contact_point_manager.h"

namespace Physika{

template <typename Scalar,int Dim>
CollidableObject<Scalar, Dim>::CollidableObject()
{

}

template <typename Scalar,int Dim>
CollidableObject<Scalar, Dim>::~CollidableObject()
{

}

template <typename Scalar,int Dim>
bool CollidableObject<Scalar, Dim>::collideWithObject(CollidableObject<Scalar, Dim> *object, Vector<Scalar,Dim> &contact_point, Vector<Scalar,Dim> &contact_normal, CollisionDetectionMethod<Scalar, Dim>* method)
{
    bool need_delete = false;
    if(method == NULL)
    {
        method = new CollisionDetectionMethodDTBVH<Scalar, Dim>;
        need_delete = true;
    }
    method->addCollidableObject(this);
    method->addCollidableObject(object);
    bool is_collide = method->collisionDetection();
    if(is_collide)
    {
        Vector<Scalar, Dim> position(0), normal(0);
        for(unsigned int i = 0; i < method->numContactPoint(); ++i)
        {
            position += method->contactPoint(i)->globalContactPosition();
            normal += method->contactPoint(i)->globalContactNormalLhs();
        }
        if(method->numContactPoint() != 0)
        {
            position /= method->numContactPoint();
            normal.normalize();
        }
        contact_point = position;
        contact_normal = normal;
    }
    if(need_delete)
        delete method;
    return is_collide;
}

template class CollidableObject<float, 2>;
template class CollidableObject<double, 2>;
template class CollidableObject<float, 3>;
template class CollidableObject<double, 3>;

}
