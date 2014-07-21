/*
 * @file rigid_response_method.cpp
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

#include <stdio.h>
#include "Physika_Dynamics/Rigid_Body/rigid_response_method.h"
#include "Physika_Dynamics/Rigid_Body/rigid_body_driver.h"

namespace Physika{

template <typename Scalar, int Dim>
RigidResponseMethod<Scalar, Dim>::RigidResponseMethod():
    rigid_driver_(NULL)
{

}


template <typename Scalar, int Dim>
RigidResponseMethod<Scalar, Dim>::~RigidResponseMethod()
{

}

template <typename Scalar, int Dim>
void RigidResponseMethod<Scalar, Dim>::setRigidDriver(RigidBodyDriver<Scalar, Dim>* rigid_driver)
{
    rigid_driver_ = rigid_driver;
}

template class RigidResponseMethod<float, 2>;
template class RigidResponseMethod<double, 2>;
template class RigidResponseMethod<float, 3>;
template class RigidResponseMethod<double, 3>;

}