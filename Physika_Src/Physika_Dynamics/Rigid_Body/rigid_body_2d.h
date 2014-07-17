/*
 * @file rigid_body_2d.h 
 * @2D rigid_body class.
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

#ifndef PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_BODY_2D_H_
#define PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_BODY_2D_H_

#include "Physika_Dynamics/Rigid_Body/rigid_body.h"
#include "Physika_Dynamics/Collidable_Objects/polygon_based_collidable_object.h"

namespace Physika{

template <typename Scalar,int Dim> class Vector;
template <typename Scalar> class Polygon;

template <typename Scalar>
class RigidBody<Scalar, 2>
{
public:
    //constructors && deconstructors
    RigidBody();
    virtual ~RigidBody();

    //get & set
    inline typename CollidableObjectInternal::ObjectType objectType() const {return object_type_;};
    inline const Transform<Scalar>& transform() const {return transform_;};
    inline Transform<Scalar>& transform() {return transform_;};
    inline void setFixed(bool is_fixed) {is_fixed_ = is_fixed;};
    inline bool isFixed() const {return is_fixed_;};
    inline Scalar density() const {return density_;};
    inline Scalar mass() const {return mass_;};

    //dynamics
    void update(Scalar dt);//update its configuration and velocity
    void performGravity(Scalar gravity, Scalar dt);//Attention! This will change its velocity

protected:
    //basic properties of a rigid body

    //can be set by public functions
    typename CollidableObjectInternal::ObjectType object_type_;
    Polygon<Scalar>* polygon_;
    Transform<Scalar> transform_;
    Scalar density_;
    bool is_fixed_;

    //obtained by internal computation
    Scalar mass_;
};
}

#endif //PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_BODY_2D_H_