/*
 * @file rigid_driver_plugin.h 
 * @Basic class for plugins of rigid body driver.
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

#ifndef PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_DRIVER_PLUGIN_H_
#define PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_DRIVER_PLUGIN_H_

#include "Physika_Dynamics/Driver/driver_plugin_base.h"

namespace Physika{

template <typename Scalar,int Dim> class RigidBody;
template <typename Scalar,int Dim> class RigidBodyDriver;

template <typename Scalar,int Dim>
class RigidDriverPlugin: public DriverPluginBase<Scalar>
{
public:
	//constructors && deconstructors
	RigidDriverPlugin();
	virtual ~RigidDriverPlugin();

	//functions called in driver
    virtual void onBeginRigidStep(int step, Scalar dt) = 0;//replace the original onBeginTimeStep in rigid body simulation
    virtual void onEndRigidStep(int step, Scalar dt) = 0;//replace the original onEndTimeStep in rigid body simulation

	virtual void onAddRigidBody(RigidBody<Scalar, Dim>* rigid_body) = 0;
	virtual void onCollisionDetection() = 0;

	//basic function
	virtual RigidBodyDriver<Scalar, Dim>* rigidDriver();
	virtual void setDriver(DriverBase<Scalar>* driver);

protected:
	RigidBodyDriver<Scalar, Dim>* rigid_driver_;
};

} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_DRIVER_PLUGIN_H_