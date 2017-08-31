/*
 * @file fem_plugin_base.cpp 
 * @brief base class of plugins for FEMSolid drivers.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Dynamics/Driver/driver_base.h"
#include "Physika_Dynamics/FEM/fem_solid.h"
#include "Physika_Dynamics/FEM/FEM_Plugins/fem_solid_plugin_base.h"

namespace Physika{

template <typename Scalar, int Dim>
FEMSolidPluginBase<Scalar,Dim>::FEMSolidPluginBase()
    :DriverPluginBase<Scalar>()
{
}

template <typename Scalar, int Dim>
FEMSolidPluginBase<Scalar,Dim>::~FEMSolidPluginBase()
{
}

template <typename Scalar, int Dim>
FEMSolid<Scalar,Dim>* FEMSolidPluginBase<Scalar,Dim>::driver()
{
    return dynamic_cast<FEMSolid<Scalar,Dim>*>(this->driver_);
}

template <typename Scalar, int Dim>
void FEMSolidPluginBase<Scalar,Dim>::setDriver(DriverBase<Scalar> *driver)
{
    if(driver == NULL)
        throw PhysikaException("Error: NULL driver pointer provided to driver plugin!");
    if(dynamic_cast<FEMSolid<Scalar,Dim>*>(driver) == NULL)
        throw PhysikaException("Wrong type of driver specified!");
    this->driver_ = driver;
}

//explicit instantiations
template class FEMSolidPluginBase<float,2>;
template class FEMSolidPluginBase<float,3>;
template class FEMSolidPluginBase<double,2>;
template class FEMSolidPluginBase<double,3>;

}  //end of namespace Physika
