/*
 * @file mpm_solid_step_method_USL.cpp 
 * @Brief the USL (update stress last) method, the stress state of the particles are
 *        updated at the end of each time step.
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

#include <cstdlib>
#include <iostream>
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Dynamics/MPM/mpm_solid_base.h"
#include "Physika_Dynamics/MPM/MPM_Step_Methods/mpm_solid_step_method_USL.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMSolidStepMethodUSL<Scalar,Dim>::MPMSolidStepMethodUSL()
    :MPMStepMethod<Scalar,Dim>()
{
}

template <typename Scalar, int Dim>
MPMSolidStepMethodUSL<Scalar,Dim>::~MPMSolidStepMethodUSL()
{
}

template <typename Scalar, int Dim>
void MPMSolidStepMethodUSL<Scalar,Dim>::advanceStep(Scalar dt)
{
    MPMSolidBase<Scalar,Dim> *mpm_solid_driver = dynamic_cast<MPMSolidBase<Scalar,Dim>*>(this->mpm_driver_);
    if(mpm_solid_driver==NULL)
        throw PhysikaException("Error: MPM driver and step method mismatch!");
    //now advance step, the constitutive model state 
    //of the particles are updated at the end of time step
    mpm_solid_driver->rasterize();
    mpm_solid_driver->solveOnGrid(dt);
    mpm_solid_driver->resolveContactOnGrid(dt);
    mpm_solid_driver->updateParticleVelocity();
    mpm_solid_driver->applyExternalForceOnParticles(dt);
    mpm_solid_driver->resolveContactOnParticles(dt);
    mpm_solid_driver->updateParticlePosition(dt);
    mpm_solid_driver->updateParticleInterpolationWeight();
    mpm_solid_driver->updateParticleConstitutiveModelState(dt);
}

//explicit instantiations
template class MPMSolidStepMethodUSL<float,2>;
template class MPMSolidStepMethodUSL<double,2>;
template class MPMSolidStepMethodUSL<float,3>;
template class MPMSolidStepMethodUSL<double,3>;

}  //end of namespace Physika
