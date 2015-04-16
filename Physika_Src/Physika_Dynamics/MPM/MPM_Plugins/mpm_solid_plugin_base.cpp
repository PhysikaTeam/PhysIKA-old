/*
 * @file mpm_solid_plugin_base.cpp 
 * @brief base class of plugins for drivers derived from MPMSolidBase.
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

#include <cstdlib>
#include <iostream>
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Dynamics/MPM/MPM_Plugins/mpm_solid_plugin_base.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMSolidPluginBase<Scalar,Dim>::MPMSolidPluginBase()
    :MPMPluginBase<Scalar,Dim>()
{
}

template <typename Scalar, int Dim>
MPMSolidPluginBase<Scalar,Dim>::~MPMSolidPluginBase()
{
}

template <typename Scalar, int Dim>
MPMSolidBase<Scalar,Dim>* MPMSolidPluginBase<Scalar,Dim>::driver()
{
    return dynamic_cast<MPMSolidBase<Scalar,Dim>*>(this->driver_);
}

template <typename Scalar, int Dim>
void MPMSolidPluginBase<Scalar,Dim>::setDriver(DriverBase<Scalar>* driver)
{
    if(driver==NULL)
        throw PhysikaException("Error: NULL driver pointer provided to driver plugin!");
    if(dynamic_cast<MPMSolidBase<Scalar,Dim>*>(driver)==NULL)
        throw PhysikaException("Wrong type of driver specified!");
    this->driver_ = driver;
}

//explicit instantiations
template class MPMSolidPluginBase<float,2>;
template class MPMSolidPluginBase<float,3>;
template class MPMSolidPluginBase<double,2>;
template class MPMSolidPluginBase<double,3>;

}  //end of namespace Physika
