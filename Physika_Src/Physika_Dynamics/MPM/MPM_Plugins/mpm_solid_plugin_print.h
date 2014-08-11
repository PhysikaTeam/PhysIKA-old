/*
 * @file mpm_solid_plugin_print.h 
 * @brief plugin to print information on screen for drivers derived from MPMSolidBase.
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

#ifndef PHYSIKA_DYNAMICS_MPM_MPM_PLUGINS_MPM_SOLID_PLUGIN_PRINT_H_
#define PHYSIKA_DYNAMICS_MPM_MPM_PLUGINS_MPM_SOLID_PLUGIN_PRINT_H_

#include "Physika_Dynamics/MPM/MPM_Plugins/mpm_solid_plugin_base.h"

namespace Physika{

template <typename Scalar, int Dim>
class MPMSolidPluginPrint: public MPMSolidPluginBase<Scalar,Dim>
{
public:
    MPMSolidPluginPrint();
    ~MPMSolidPluginPrint();

    //inherited virtual methods
    virtual void onBeginFrame(unsigned int frame);
    virtual void onEndFrame(unsigned int frame);
    virtual void onBeginTimeStep(Scalar time, Scalar dt);
    virtual void onEndTimeStep(Scalar time, Scalar dt);

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
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_MPM_PLUGINS_MPM_SOLID_PLUGIN_PRINT_H_
