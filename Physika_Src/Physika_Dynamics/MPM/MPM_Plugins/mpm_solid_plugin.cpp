/*
 * @file mpm_solid_plugin.cpp 
 * @brief base class of plugins for drivers derived from MPMSolidBase.
 * @author Tianxiang Zhang, Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Dynamics/MPM/mpm_solid_base.h"
#include "Physika_Dynamics/MPM/MPM_Plugins/mpm_solid_plugin.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMSolidPlugin<Scalar,Dim>::MPMSolidPlugin()
    :DriverPluginBase<Scalar>()
{
}

template <typename Scalar, int Dim>
MPMSolidPlugin<Scalar,Dim>::~MPMSolidPlugin()
{
}

template <typename Scalar, int Dim>
MPMSolidBase<Scalar,Dim>* MPMSolidPlugin<Scalar,Dim>::driver()
{
    return dynamic_cast<MPMSolidBase<Scalar,Dim>*>(this->driver_);
}

//explicit instantiations
template class MPMSolidPlugin<float,2>;
template class MPMSolidPlugin<float,3>;
template class MPMSolidPlugin<double,2>;
template class MPMSolidPlugin<double,3>;

}  //end of namespace Physika
