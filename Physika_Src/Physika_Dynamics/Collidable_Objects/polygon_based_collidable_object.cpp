/*
 * @file  polygon_based_collidable_object.cpp
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

#include "Physika_Core/Transform/transform_2d.h"
#include "Physika_Dynamics/Collidable_Objects/polygon_based_collidable_object.h"

namespace Physika{

template <typename Scalar>
PolygonBasedCollidableObject<Scalar>::PolygonBasedCollidableObject():
    polygon_(NULL),
    transform_(NULL)
{
}

template <typename Scalar>
PolygonBasedCollidableObject<Scalar>::~PolygonBasedCollidableObject()
{
}

template <typename Scalar>
typename CollidableObjectInternal::ObjectType PolygonBasedCollidableObject<Scalar>::objectType() const
{
    return CollidableObjectInternal::POLYGON;
}

//explicit instantitation
template class PolygonBasedCollidableObject<float>;
template class PolygonBasedCollidableObject<double>;

}