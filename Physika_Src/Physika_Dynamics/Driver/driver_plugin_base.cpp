/*
 * @file driver_plugin_base.cpp
 * @Basic class for plugins of a simulation driver.
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

#include "Physika_Dynamics/Driver/driver_base.h"
#include "Physika_Dynamics/Driver/driver_plugin_base.h"

namespace Physika{

template <typename Scalar>
DriverPluginBase<Scalar>::DriverPluginBase():
    driver_(NULL)
{

}

template <typename Scalar>
DriverPluginBase<Scalar>::~DriverPluginBase()
{

}

template <typename Scalar>
DriverBase<Scalar>* DriverPluginBase<Scalar>::driver()
{
    return driver_;
}

template <typename Scalar>
void DriverPluginBase<Scalar>::setDriver(DriverBase<Scalar>* driver)
{
    driver_ = driver;
}

//explicit instantiation
template class DriverPluginBase<float>;
template class DriverPluginBase<double>;

}
