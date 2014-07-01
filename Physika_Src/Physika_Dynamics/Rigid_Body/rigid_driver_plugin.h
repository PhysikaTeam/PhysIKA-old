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

namespace Physika{

template <typename Scalar,int Dim> class RigidBody;
template <typename Scalar,int Dim> class RigidBodyDriver;

template <typename Scalar,int Dim>
class RigidDriverPlugin
{
public:
	//constructors && deconstructors
	RigidDriverPlugin();
	virtual ~RigidDriverPlugin();

	//functions called in driver
	virtual void onRun() = 0;
	virtual void onAdvanceFrame() = 0;
	virtual void onInitialize() = 0;
	virtual void onAdvanceStep(Scalar dt) = 0;
	virtual void onAddRigidBody(RigidBody<Scalar, Dim>* rigid_body) = 0;

	//basic function
	virtual RigidBodyDriver<Scalar, Dim>* driver();
	virtual void setDriver(RigidBodyDriver<Scalar, Dim>* driver);

protected:
	RigidBodyDriver<Scalar, Dim>* driver_;
};

} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_DRIVER_PLUGIN_H_