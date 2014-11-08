/*
 * @file mpm_solid_step_method_USF.cpp 
 * @Brief the USF (update stress first) method, the stress state of the particles are
 *        updated at the beginning of each time step.
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


#include <cstdlib>
#include <iostream>
#include "Physika_Dynamics/MPM/mpm_solid_base.h"
#include "Physika_Dynamics/MPM/MPM_Step_Methods/mpm_solid_step_method_USF.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMSolidStepMethodUSF<Scalar,Dim>::MPMSolidStepMethodUSF()
    :MPMStepMethod<Scalar,Dim>()
{
}

template <typename Scalar, int Dim>
MPMSolidStepMethodUSF<Scalar,Dim>::~MPMSolidStepMethodUSF()
{
}

template <typename Scalar, int Dim>
void MPMSolidStepMethodUSF<Scalar,Dim>::advanceStep(Scalar dt)
{
    MPMSolidBase<Scalar,Dim> *mpm_solid_driver = dynamic_cast<MPMSolidBase<Scalar,Dim>*>(this->mpm_driver_);
    if(mpm_solid_driver==NULL)
    {
        std::cerr<<"Error: MPM driver and step method mismatch, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    //now advance step, the constitutive model state 
    //of the particles are updated at beginning of time step
    mpm_solid_driver->rasterize();
    mpm_solid_driver->updateParticleConstitutiveModelState(dt);
    mpm_solid_driver->solveOnGrid(dt);
    mpm_solid_driver->resolveContactOnGrid(dt);
    mpm_solid_driver->updateParticleVelocity();
    mpm_solid_driver->applyExternalForceOnParticles(dt);
    mpm_solid_driver->resolveContactOnParticles(dt);
    mpm_solid_driver->updateParticlePosition(dt);
    mpm_solid_driver->updateParticleInterpolationWeight();
}

//explicit instantiations
template class MPMSolidStepMethodUSF<float,2>;
template class MPMSolidStepMethodUSF<double,2>;
template class MPMSolidStepMethodUSF<float,3>;
template class MPMSolidStepMethodUSF<double,3>;

}  //end of namespace Physika
