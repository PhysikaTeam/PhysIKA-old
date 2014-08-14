/*
 * @file  collidable_object.h
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

#ifndef PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_COLLIDABLE_OBJECT_H_
#define PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_COLLIDABLE_OBJECT_H_

#include <cstddef>

namespace Physika{

template <typename Scalar,int Dim> class Vector;
template <typename Scalar,int Dim> class CollisionDetectionMethod;

namespace CollidableObjectInternal{
    enum ObjectType {MESH_BASED, IMPLICIT, POLYGON};
}

template <typename Scalar,int Dim>
class CollidableObject
{
public:
	//constructors && deconstructors
    CollidableObject();
    virtual ~CollidableObject();

	//get & set
	virtual CollidableObjectInternal::ObjectType objectType() const=0;

    //given position of a point, detect collision with the collidable object. contact_normal will be modified after calling this function
    virtual bool collideWithPoint(Vector<Scalar,Dim> *point, Vector<Scalar,Dim> &contact_normal) = 0;

    //given another collidable, detect collision with this collidable object. contact_point and contact_normal will be modified after calling this function
    virtual bool collideWithObject(CollidableObject<Scalar, Dim> *object, Vector<Scalar,Dim> &contact_point, Vector<Scalar,Dim> &contact_normal, CollisionDetectionMethod<Scalar, Dim>* method = NULL);
protected:
	
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_COLLIDABLE_OBJECT_H_
