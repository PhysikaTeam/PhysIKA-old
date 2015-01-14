/*
 * @file collidable_object.h 
 * @brief base class of all collidable objects. A collidable object is the kinematic object
 *            in scene that is used for simple contact effects.
 * @author Fei Zhu
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

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

template <typename Scalar, int Dim>
class CollidableObject
{
public:
    CollidableObject();
    CollidableObject(Scalar mu, bool sticky);
    CollidableObject(const CollidableObject<Scalar,Dim> &object);
    virtual ~CollidableObject();
    CollidableObject<Scalar,Dim>& operator= (const CollidableObject<Scalar,Dim> &object);
    virtual CollidableObject<Scalar,Dim>* clone() const = 0;
    //input a point and the velocity at the point, return true and the corresponding impulse enforced at the point due to contact if the point collides
    //with the collidable object, else return false
    virtual bool collide(const Vector<Scalar,Dim> &point, const Vector<Scalar,Dim> &velocity, Vector<Scalar,Dim> &velocity_impulse) const = 0;
    virtual Scalar distance(const Vector<Scalar,Dim> &point) const = 0; //distance to the surface of the object
    virtual Scalar signedDistance(const Vector<Scalar,Dim> &point) const = 0;  //signed distance to the surface of the object
    virtual Vector<Scalar,Dim> normal(const Vector<Scalar,Dim> &point) const = 0; //normal at given point
    Scalar mu() const;
    void setMu(Scalar mu);
    bool isSticky() const;
    void setSticky(bool sticky);
    Scalar collideThreshold() const;
    void setCollideThreshold(Scalar threshold);
    Vector<Scalar,Dim> velocity() const;
    void setVelocity(const Vector<Scalar,Dim> &velocity);
protected:
    Scalar mu_; //coefficient of friction
    bool sticky_; //the contact with the object is sticky
    Scalar collide_threshold_; //the distance threshold that can be considered as collide
    Vector<Scalar,Dim> velocity_;
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_COLLIDABLE_OBJECT_H_
