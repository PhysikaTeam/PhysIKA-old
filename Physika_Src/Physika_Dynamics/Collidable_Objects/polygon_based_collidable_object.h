/*
 * @file  polygon_based_collidable_object.h
 * @2D collidable object based on the polygon
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

#ifndef PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_POLYGON_BASED_COLLIDABLE_OBJECT_H_
#define PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_POLYGON_BASED_COLLIDABLE_OBJECT_H_

#include "Physika_Dynamics/Collidable_Objects/collidable_object.h"

namespace Physika{

template <typename Scalar> class Polygon;

template <typename Scalar>
class PolygonBasedCollidableObject : public CollidableObject<Scalar, 2>
{
public:
    PolygonBasedCollidableObject();
    ~PolygonBasedCollidableObject();
    typename CollidableObjectInternal::ObjectType objectType() const;
protected:
    Polygon<Scalar>* polygon_;
    Transform<Scalar>* transform_;
};

}

#endif //PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_POLYGON_BASED_COLLIDABLE_OBJECT_H_