/*
 * @file mpm_solid_plugin_base.h 
 * @brief base class of plugins for drivers derived from MPMSolidBase.
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

#ifndef PHYSIKA_DYNAMICS_MPM_MPM_PLUGINS_MPM_SOLID_PLUGIN_BASE_H_
#define PHYSIKA_DYNAMICS_MPM_MPM_PLUGINS_MPM_SOLID_PLUGIN_BASE_H_

#include "Physika_Dynamics/MPM/mpm_solid_base.h"
#include "Physika_Dynamics/MPM/MPM_Plugins/mpm_plugin_base.h"

namespace Physika{

template <typename Scalar, int Dim>
class MPMSolidPluginBase: public MPMPluginBase<Scalar,Dim>
{
public:
    MPMSolidPluginBase();
    virtual ~MPMSolidPluginBase();

    //inherited virtual methods
    virtual void onBeginFrame(unsigned int frame) = 0;
    virtual void onEndFrame(unsigned int frame) = 0;
    virtual void onBeginTimeStep(Scalar time, Scalar dt) = 0;
    virtual void onEndTimeStep(Scalar time, Scalar dt) = 0;
    virtual MPMSolidBase<Scalar,Dim>* driver();
    virtual void setDriver(DriverBase<Scalar>* driver);

    //MPM Solid driver specific virtual methods
    virtual void onRasterize() = 0;
    virtual void onSolveOnGrid(Scalar dt) = 0;
    virtual void onResolveContactOnGrid(Scalar dt) = 0;
    virtual void onResolveContactOnParticles(Scalar dt) = 0;
    virtual void onUpdateParticleInterpolationWeight() = 0;
    virtual void onUpdateParticleConstitutiveModelState(Scalar dt) = 0;
    virtual void onUpdateParticleVelocity() = 0;
    virtual void onApplyExternalForceOnParticles(Scalar dt) = 0;
    virtual void onUpdateParticlePosition(Scalar dt) = 0;
protected:
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_MPM_PLUGINS_MPM_SOLID_PLUGIN_BASE_H_
