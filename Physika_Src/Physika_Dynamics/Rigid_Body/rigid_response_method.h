/*
 * @file rigid_response_method.h 
 * @Base class of rigid-body collision response methods
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

#ifndef PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_RESPONSE_METHOD_H_
#define PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_RESPONSE_METHOD_H_

namespace Physika{

template <typename Scalar,int Dim> class RigidBodyDriver;

template <typename Scalar,int Dim>
class RigidResponseMethod
{
public:
     //constructor
    RigidResponseMethod();
    virtual ~RigidResponseMethod();

    //dynamic function used in a driver
    virtual void collisionResponse() = 0;

    //set
    void setRigidDriver(RigidBodyDriver<Scalar, Dim>* rigid_driver);

protected:
    RigidBodyDriver<Scalar, Dim>* rigid_driver_;
};

}

#endif //PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_RESPONSE_METHOD_H_