/*
 * @file PDM_plugin_base.cpp 
 * @brief base class of plugins for PDM drivers.
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Dynamics/PDM/PDM_Plugins/PDM_plugin_base.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMPluginBase<Scalar, Dim>::PDMPluginBase()
	:DriverPluginBase<Scalar>()
{

}

template <typename Scalar, int Dim>
PDMPluginBase<Scalar, Dim>::~PDMPluginBase()
{

}

template <typename Scalar, int Dim>
PDMBase<Scalar, Dim> * PDMPluginBase<Scalar, Dim>::driver()
{
	return dynamic_cast<PDMBase<Scalar, Dim>*>(this->driver_);
}

template <typename Scalar, int Dim>
void PDMPluginBase<Scalar, Dim>::setDriver(DriverBase<Scalar>* driver)
{
	if(driver == NULL)
	{
		std::cerr<<"Error: NULL driver pointer provided to driver plugins, program abort!\n";
		std::exit(EXIT_FAILURE);
	}
	this->driver_ = driver;
}

// explicit instantiations
template class PDMPluginBase<float, 2>;
template class PDMPluginBase<float, 3>;
template class PDMPluginBase<double, 2>;
template class PDMPluginBase<double, 3>;


}