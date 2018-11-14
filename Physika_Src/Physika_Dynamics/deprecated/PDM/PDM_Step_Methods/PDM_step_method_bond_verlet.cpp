/*
 * @file PDM_step_method_bond_Verlet.h 
 * @Basic PDMStepMethodBondVerlet class. Verlet step method class, a Verlet explicit step method implemented
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

#include <numeric>
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"

#include "Physika_Dynamics/PDM/PDM_base.h"
#include "Physika_Dynamics/PDM/PDM_particle.h"

#include "Physika_Dynamics/PDM/PDM_step_methods/PDM_step_method_bond_Verlet.h"

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Fracture_Methods/PDM_fracture_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Impact_Methods/PDM_impact_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Topology_Control_Methods/PDM_topology_control_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Boundary_Condition_Methods/PDM_boundary_condition_method.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMStepMethodBondVerlet<Scalar, Dim>::PDMStepMethodBondVerlet()
    :PDMStepMethodBond()
{

}

template <typename Scalar, int Dim>
PDMStepMethodBondVerlet<Scalar, Dim>::PDMStepMethodBondVerlet(PDMBond<Scalar, Dim> * pdm_base)
    :PDMStepMethodBond(pdm_base)
{

}

template <typename Scalar, int Dim>
PDMStepMethodBondVerlet<Scalar, Dim>::~PDMStepMethodBondVerlet()
{

}

template <typename Scalar, int Dim>
void PDMStepMethodBondVerlet<Scalar, Dim>::advanceStep(Scalar dt)
{
    Timer timer;

    // update family parameters
    timer.startTimer();
    this->updateParticleFamilyParameters();
    timer.stopTimer();
    std::cout<<"time for update family parameters: "<<timer.getElapsedTime()<<std::endl;

    // for more detail about Verlet method, please refer to SCA14. 
    unsigned int num_particle = this->pdm_base_->numSimParticles();

    //update displacement and half velocity
    for (unsigned int par_k = 0; par_k < num_particle; par_k++)
    {
        PDMParticle<Scalar, Dim> & particle_k = this->pdm_base_->particle(par_k);
        Vector<Scalar, Dim> delta_vel = (0.5*dt/particle_k.mass())*this->pdm_base_->particleForce(par_k);
        this->pdm_base_->addParticleVelocity(par_k, delta_vel);
        this->pdm_base_->addParticleDisplacement(par_k, particle_k.velocity()*dt);
    }

    // particles forces have to be reset
    this->pdm_base_->resetParticleForce();

    if (enable_collision_)
    {
        if (collision_method_ == NULL)
        {
            std::cerr<<"error: collision method not specified!\n";
            std::exit(EXIT_FAILURE);
        }
        this->collision_method_->collisionMethod();
    }

    // gravity
    this->addGravity();

    // boundary: extra force
    if (enable_boundary_condition_)
    {
        if (boundary_condition_method_ == NULL)
        {
            std::cerr<<"error: boundary condition method not specified!\n";
            std::exit(EXIT_FAILURE);
        }
        this->boundary_condition_method_->setSpecifiedParticleExtraForce();
    }

    // calculate Force
    this->calculateForce(dt);

    // impact: its position needs further consideration
    if (this->enable_impact_)
    {
        if (impact_method_ == NULL)
        {
            std::cerr<<"error: impact method not specified!\n";
            std::exit(EXIT_FAILURE);
        }
        this->impact_method_->applyImpact(dt);
    }

    //update total velocity
    for(unsigned int par_k = 0; par_k < num_particle; par_k++)
    {
        PDMParticle<Scalar, Dim> & particle_k = this->pdm_base_->particle(par_k);
        Vector<Scalar, Dim> delta_vel = (0.5*dt/particle_k.mass())*this->pdm_base_->particleForce(par_k);
        this->pdm_base_->addParticleVelocity(par_k, delta_vel);
    }

    // boundary
    if (enable_boundary_condition_)
    {
        this->boundary_condition_method_->resetSpecifiedParticlePos();
        this->boundary_condition_method_->resetSpecifiedParticleVel();
        this->boundary_condition_method_->setSpecifiedParticleVel();
        this->boundary_condition_method_->exertFloorBoundaryCondition();
    }

    // topology control, used to split the tet elements
    if (enable_topology_control_)
    {
        if (this->topology_control_method_ == NULL)
        {
            std::cerr<<"error: topology control method not specified!\n";
            std::exit(EXIT_FAILURE);
        }
        this->topology_control_method_->topologyControlMethod(dt);
    }

    std::cout<<"particle dis(par_id = 0): "<<this->pdm_base_->particleDisplacement(0)<<std::endl;
    std::cout<<"particle for(par_id = 0): "<<this->pdm_base_->particleForce(0)<<std::endl;
    std::cout<<"particle vel(par_id = 0): "<<this->pdm_base_->particleVelocity(0)<<std::endl;
}


//explicit instantiations
template class PDMStepMethodBondVerlet<float, 3>;
template class PDMStepMethodBondVerlet<double, 3>;

}