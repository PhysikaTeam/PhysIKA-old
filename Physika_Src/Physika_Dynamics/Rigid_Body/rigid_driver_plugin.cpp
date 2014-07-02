/*
 * @file rigid_driver_plugin.cpp
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

#include "Physika_Dynamics/Rigid_Body/rigid_driver_plugin.h"
#include "Physika_Dynamics/Rigid_Body/rigid_body_driver.h"

namespace Physika{

template <typename Scalar,int Dim>
RigidDriverPlugin<Scalar, Dim>::RigidDriverPlugin():
	rigid_driver_(NULL)
{

}

template <typename Scalar,int Dim>
RigidDriverPlugin<Scalar, Dim>::~RigidDriverPlugin()
{

}

template <typename Scalar,int Dim>
RigidBodyDriver<Scalar, Dim>* RigidDriverPlugin<Scalar, Dim>::rigidDriver()
{
	return rigid_driver_;
}

template <typename Scalar,int Dim>
void RigidDriverPlugin<Scalar, Dim>::setDriver(DriverBase<Scalar>* driver)
{
	rigid_driver_ = dynamic_cast<RigidBodyDriver<Scalar, Dim>*>(driver);
	if(rigid_driver_ != NULL)
		this->driver_ = driver;
}

//explicit instantiation
template class RigidDriverPlugin<float, 3>;
template class RigidDriverPlugin<double, 3>;

}