/*
 * @file mpm_solid_plugin.h 
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

#ifndef PHYSIKA_DYNAMICS_MPM_MPM_PLUGINS_MPM_SOLID_PLUGIN_H_
#define PHYSIKA_DYNAMICS_MPM_MPM_PLUGINS_MPM_SOLID_PLUGIN_H_

#include "Physika_Dynamics/Driver/driver_plugin_base.h"

namespace Physika{

template <typename Scalar, int Dim> class MPMSolidBase;

template <typename Scalar, int Dim>
class MPMSolidPlugin: public DriverPluginBase<Scalar>
{
public:
    MPMSolidPlugin();
    virtual ~MPMSolidPlugin();

    //inherited virtual methods
    virtual void onBeginFrame(unsigned int frame) = 0;
    virtual void onEndFrame(unsigned int frame) = 0;
    virtual void onBeginTimeStep(Scalar time, Scalar dt) = 0;
    virtual void onEndTimeStep(Scalar time, Scalar dt) = 0;
    virtual MPMSolidBase<Scalar,Dim>* driver();

protected:
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_MPM_PLUGINS_MPM_SOLID_PLUGIN_H_
