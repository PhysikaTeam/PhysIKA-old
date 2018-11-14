/*
* @file sph_plugin_base.h
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

#ifndef PHYSIKA_DYNAMICS_SPH_SPH_PLUGIN_BASE_H_
#define PHYSIKA_DYNAMICS_SPH_SPH_PLUGIN_BASE_H_

#include "Physika_Dynamics/Driver/driver_plugin_base.h"

namespace Physika{

template<typename Scalar, int Dim>
class SPHBase;

template<typename Scalar, int Dim>
class SPHPluginBase: public DriverPluginBase<Scalar>
{
public:

    SPHPluginBase();
    virtual ~SPHPluginBase();

    //inherited virtual methods
    virtual void onBeginFrame(unsigned int frame) = 0;
    virtual void onEndFrame(unsigned int frame) = 0;
    virtual void onBeginTimeStep(Scalar time, Scalar dt) = 0;
    virtual void onEndTimeStep(Scalar time, Scalar dt) = 0;

    virtual SPHBase<Scalar, Dim> * driver();
    virtual void setDriver(DriverBase<Scalar>* driver);

};

}//end of namespace Physika

#endif //PHYSIKA_DYNAMICS_SPH_SPH_PLUGIN_BASE_H_