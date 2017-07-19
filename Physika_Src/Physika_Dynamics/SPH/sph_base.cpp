/*
 * @file sph_base.cpp
 * @Brief class SPHBase
 * @author Wei Chen
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

#include "Physika_Dynamics/Driver/driver_plugin_base.h"
#include "Physika_Dynamics/SPH/sph_plugin_base.h"

#include "Physika_Dynamics/SPH/sph_base.h"


namespace Physika{

template<typename Scalar, int Dim>
SPHBase<Scalar, Dim>::SPHBase()
{

}


template<typename Scalar, int Dim>
SPHBase<Scalar, Dim>::SPHBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :DriverBase(start_frame, end_frame, frame_rate, max_dt, write_to_file)
{

}

template<typename Scalar, int Dim>
SPHBase<Scalar, Dim>::~SPHBase()
{

}

template<typename Scalar, int Dim>
void SPHBase<Scalar, Dim>::initConfiguration(const std::string &file_name)
{
    throw PhysikaException("Error: function initConfiguration not implemented!\n");
}

template<typename Scalar, int Dim>
void SPHBase<Scalar, Dim>::printConfigFileFormat()
{
    throw PhysikaException("Error: function printConfigFileFormat not implemented!\n");
}

template<typename Scalar, int Dim>
void SPHBase<Scalar, Dim>::initSimulationData()
{
    //do nothing
}

template<typename Scalar, int Dim>
Scalar SPHBase<Scalar, Dim>::computeTimeStep()
{
    //to do
    return this->max_dt_;
}

template<typename Scalar, int Dim>
void SPHBase<Scalar, Dim>::addPlugin(DriverPluginBase<Scalar>* plugin)
{
    if (plugin == nullptr)
        throw PhysikaException("Error: null plugin provided");

    if (dynamic_cast<SPHPluginBase<Scalar, Dim> *>(plugin) == nullptr)
        throw PhysikaException("Error: wrong type of plugin provided!\n");

    plugin->setDriver(this);
    this->plugins_.push_back(plugin);
}

template<typename Scalar, int Dim>
bool SPHBase<Scalar, Dim>::withRestartSupport() const
{
    return false;
}

template<typename Scalar, int Dim>
void SPHBase<Scalar, Dim>::write(const std::string &file_name)
{
    throw PhysikaException("Error: function write not implemented!\n");
}

template<typename Scalar, int Dim>
void SPHBase<Scalar, Dim>::read(const std::string &file_name)
{
    throw PhysikaException("Error: function read not implemented!\n");
}

//explicit instantiations
template class SPHBase<float, 2>;
template class SPHBase<float, 3>;
template class SPHBase<double, 2>;
template class SPHBase<double, 3>;

}//end of namespace Physika