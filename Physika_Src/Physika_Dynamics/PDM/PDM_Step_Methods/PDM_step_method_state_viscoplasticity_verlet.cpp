/*
 * @file PDM_step_method_state_viscoplasticiy_verlet.cpp
 * @Basic PDMStepMethodStateViscoPlasticityVerlet class.
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

#include "Physika_Dynamics/PDM/PDM_state.h"
#include "Physika_Dynamics/PDM/PDM_state_2d.h"

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Fracture_Methods/PDM_fracture_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Impact_Methods/PDM_impact_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Topology_Control_Methods/PDM_topology_control_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Boundary_Condition_Methods/PDM_boundary_condition_method.h"

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_step_method_state_viscoplasticity_verlet.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMStepMethodStateViscoPlasticityVerlet<Scalar, Dim>::PDMStepMethodStateViscoPlasticityVerlet()
{

}

template <typename Scalar, int Dim>
PDMStepMethodStateViscoPlasticityVerlet<Scalar, Dim>::PDMStepMethodStateViscoPlasticityVerlet(PDMState<Scalar, Dim> * pdm_base)
    :PDMStepMethodStateViscoPlasticity(pdm_base)
{

}

template <typename Scalar, int Dim>
PDMStepMethodStateViscoPlasticityVerlet<Scalar, Dim>::~PDMStepMethodStateViscoPlasticityVerlet()
{

}

template <typename Scalar, int Dim>
void PDMStepMethodStateViscoPlasticityVerlet<Scalar, Dim>::advanceStep(Scalar dt)
{
    Timer timer;

    // update family parameters
    timer.startTimer();
    this->updateParticleFamilyParameters();
    timer.stopTimer();

    std::cout<<"--------------------------------------------------------------------"<<std::endl;
    std::cout<<"time for update family parameters: "<<timer.getElapsedTime()<<std::endl;
    std::cout<<"--------------------------------------------------------------------"<<std::endl;

    // for more detail about Velocity Verlet method, please refer to SCA14. 

    unsigned int num_particle = this->pdm_base_->numSimParticles();

    //update displacement and half step velocity
    for (unsigned int par_k = 0; par_k < num_particle; par_k++)
    {
        PDMParticle<Scalar, Dim> & particle_k = this->pdm_base_->particle(par_k);
        Vector<Scalar, Dim> delta_vel = (0.5*dt/particle_k.mass())*this->pdm_base_->particleForce(par_k);
        this->pdm_base_->addParticleVelocity(par_k, delta_vel);
        this->pdm_base_->addParticleDisplacement(par_k, particle_k.velocity()*dt);
    }

    // particles forces have to be reset
    this->pdm_base_->resetParticleForce();

    //collision
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

    // damping
    this->addDampingForce();

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

    //calculate force
    timer.startTimer();
    this->calculateForce(dt);
    timer.stopTimer();

    std::cout<<"--------------------------------------------------------------------"<<std::endl;
    std::cout<<"time for calculate force: "<<timer.getElapsedTime()<<std::endl;
    std::cout<<"--------------------------------------------------------------------"<<std::endl;
    
    //update total step velocity
    for(unsigned int par_k = 0; par_k < num_particle; par_k++)
    {
        PDMParticle<Scalar, Dim> & particle_k = this->pdm_base_->particle(par_k);
        Vector<Scalar, Dim> delta_vel = (0.5*dt/particle_k.mass())*this->pdm_base_->particleForce(par_k);

        this->pdm_base_->addParticleVelocity(par_k, delta_vel);
        this->pdm_base_->setParticleVelocity(par_k, (1.0-this->vel_decay_ratio_)*this->pdm_base_->particleVelocity(par_k)); 
    }
    
    // impact
    if (this->enable_impact_)
    {
        if (impact_method_ == NULL)
        {
            std::cerr<<"error: impact method not specified!\n";
            std::exit(EXIT_FAILURE);
        }
        this->impact_method_->applyImpact(dt);
    }

    // boundary
    if (enable_boundary_condition_)
    {
        this->boundary_condition_method_->resetSpecifiedParticlePos();
        this->boundary_condition_method_->resetSpecifiedParticleVel();
        this->boundary_condition_method_->setSpecifiedParticleVel();
        this->boundary_condition_method_->resetSpecifiedParticleXYZVel();
        this->boundary_condition_method_->exertFloorBoundaryCondition();
    }

    // laplacian damping
    this->addLaplacianDamping();

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
template class PDMStepMethodStateViscoPlasticityVerlet<float,2>;
template class PDMStepMethodStateViscoPlasticityVerlet<double,2>;
template class PDMStepMethodStateViscoPlasticityVerlet<float,3>;
template class PDMStepMethodStateViscoPlasticityVerlet<double,3>;

}//end of namespace Physika