/*
 * @file mpm_plugin_base.cpp
 * @brief base class of plugins for MPM drivers.
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

#include "Physika_Dynamics/Driver/driver_base.h"
#include "Physika_Dynamics/MPM/MPM_Plugins/mpm_plugin_base.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMPluginBase<Scalar,Dim>::MPMPluginBase()
    :DriverPluginBase<Scalar>()
{
}

template <typename Scalar, int Dim>
MPMPluginBase<Scalar,Dim>::~MPMPluginBase()
{
}

//explicit instantiations
template class MPMPluginBase<float,2>;
template class MPMPluginBase<float,3>;
template class MPMPluginBase<double,2>;
template class MPMPluginBase<double,3>;

}  //end of namespace Physika
