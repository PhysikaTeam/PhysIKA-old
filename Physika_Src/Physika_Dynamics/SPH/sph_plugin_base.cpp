/*
 * @file sph_plugin_base.cpp
 * @brief base class of plugins for SPH drivers.
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

#include "Physika_Core/Utilities/physika_exception.h"

#include "Physika_Dynamics/Driver/driver_base.h"
#include "Physika_Dynamics/SPH/sph_base.h"
#include "Physika_Dynamics/SPH/sph_plugin_base.h"

namespace Physika{

template<typename Scalar, int Dim>
SPHPluginBase<Scalar, Dim>::SPHPluginBase()
{

}

template<typename Scalar, int Dim>
SPHPluginBase<Scalar, Dim>::~SPHPluginBase()
{

}

template<typename Scalar, int Dim>
SPHBase<Scalar, Dim> * SPHPluginBase<Scalar, Dim>::driver()
{
    return dynamic_cast<SPHBase<Scalar, Dim> *>(this->driver_);
}

template<typename Scalar, int Dim>
void SPHPluginBase<Scalar, Dim>::setDriver(DriverBase<Scalar>* driver)
{
    if (driver == nullptr)
        throw PhysikaException("Error: null driver is provided!\n");

    if (dynamic_cast<SPHBase<Scalar, Dim> *>(driver) == nullptr)
        throw PhysikaException("Error: wrong type of driver is provided!\n");

    this->driver_ = driver;
}

//explicit instantiations 
template class SPHPluginBase<float, 2>;
template class SPHPluginBase<float, 3>;
template class SPHPluginBase<double, 2>;
template class SPHPluginBase<double, 3>;

}//end of namespace Physika
