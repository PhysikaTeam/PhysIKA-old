/*
 * @file mpm_solid_plugin_render.h 
 * @brief plugin for real-time render of MPMSolid driver.
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

#ifndef PHYSIKA_DYNAMICS_MPM_MPM_PLUGINS_MPM_SOLID_PLUGIN_RENDER_H_
#define PHYSIKA_DYNAMICS_MPM_MPM_PLUGINS_MPM_SOLID_PLUGIN_RENDER_H_

#include "Physika_Dynamics/MPM/mpm_solid.h"
#include "Physika_Dynamics/MPM/MPM_Plugins/mpm_solid_plugin_base.h"

namespace Physika{

class GlutWindow;

template <typename Scalar, int Dim>
class MPMSolidPluginRender: public MPMSolidPluginBase<Scalar,Dim>
{
public:
    MPMSolidPluginRender();
    ~MPMSolidPluginRender();

    //inherited virtual methods
    virtual void onBeginFrame(unsigned int frame);
    virtual void onEndFrame(unsigned int frame);
    virtual void onBeginTimeStep(Scalar time, Scalar dt);
    virtual void onEndTimeStep(Scalar time, Scalar dt);
    virtual MPMSolid<Scalar,Dim>* driver();
    virtual void setDriver(DriverBase<Scalar> *driver);

    //MPM Solid driver specific virtual methods
    virtual void onRasterize();
    virtual void onSolveOnGrid(Scalar dt);
    virtual void onPerformGridCollision(Scalar dt);
    virtual void onPerformParticleCollision(Scalar dt);
    virtual void onUpdateParticleInterpolationWeight();
    virtual void onUpdateParticleConstitutiveModelState(Scalar dt);
    virtual void onUpdateParticleVelocity();
    virtual void onUpdateParticlePosition(Scalar dt);

protected:
    GlutWindow *window_;
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_MPM_PLUGINS_MPM_SOLID_PLUGIN_RENDER_H_
