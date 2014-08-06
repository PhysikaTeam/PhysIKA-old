/*
 * @file mpm_solid_step_method_MUSL.cpp 
 * @Brief the MUSL (modified update stress last) method
 * @reference: Application of Particle-in-Cell method to Solid Mechanics
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
#include "Physika_Dynamics/MPM/MPM_Step_Methods/mpm_solid_step_method_MUSL.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMSolidStepMethodMUSL<Scalar,Dim>::MPMSolidStepMethodMUSL()
    :MPMStepMethod<Scalar,Dim>()
{
}

template <typename Scalar, int Dim>
MPMSolidStepMethodMUSL<Scalar,Dim>::~MPMSolidStepMethodMUSL()
{
}

template <typename Scalar, int Dim>
void MPMSolidStepMethodMUSL<Scalar,Dim>::advanceStep(Scalar dt)
{
    MPMSolidBase<Scalar,Dim> *mpm_solid_driver = dynamic_cast<MPMSolidBase<Scalar,Dim>*>(this->mpm_driver_);
    if(mpm_solid_driver==NULL)
    {
        std::cerr<<"Error: MPM driver and step method mismatch, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    //now advance step, the constitutive model state 
    //of the particles are updated at the end of time step with the newly rasterized grid data
    mpm_solid_driver->rasterize();
    mpm_solid_driver->solveOnGrid(dt);
    mpm_solid_driver->performGridCollision(dt);
    mpm_solid_driver->updateParticleVelocity();
    mpm_solid_driver->performParticleCollision(dt);
    mpm_solid_driver->updateParticlePosition(dt);
    mpm_solid_driver->updateParticleInterpolationWeight();
    mpm_solid_driver->rasterize();
    mpm_solid_driver->updateParticleConstitutiveModelState(dt);
}

//explicit instantiations
template class MPMSolidStepMethodMUSL<float,2>;
template class MPMSolidStepMethodMUSL<double,2>;
template class MPMSolidStepMethodMUSL<float,3>;
template class MPMSolidStepMethodMUSL<double,3>;

}  //end of namespace Physika
