/*
 * @file PDM_step_method_state_viscoplasticiy_semi_implicit.cpp
 * @Basic PDMStepMethodStateViscoPlasticitySemiImplicit class.
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

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_step_method_state_viscoplasticity_semi_implicit.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMStepMethodStateViscoPlasticitySemiImplicit<Scalar, Dim>::PDMStepMethodStateViscoPlasticitySemiImplicit()
{

}

template <typename Scalar, int Dim>
PDMStepMethodStateViscoPlasticitySemiImplicit<Scalar, Dim>::PDMStepMethodStateViscoPlasticitySemiImplicit(PDMState<Scalar, Dim> * pdm_base)
    :PDMStepMethodStateViscoPlasticity(pdm_base)
{

}

template <typename Scalar, int Dim>
PDMStepMethodStateViscoPlasticitySemiImplicit<Scalar, Dim>::~PDMStepMethodStateViscoPlasticitySemiImplicit()
{

}

template <typename Scalar, int Dim>
void PDMStepMethodStateViscoPlasticitySemiImplicit<Scalar, Dim>::advanceStep(Scalar dt)
{
    Timer timer;

    // update family parameters
    timer.startTimer();
    this->updateParticleFamilyParameters();
    timer.stopTimer();
    std::cout<<"time for update family parameters: "<<timer.getElapsedTime()<<std::endl;

    // for more detail about Velocity Verlet method, please refer to SCA14. 
    
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
    std::cout<<"time for calculate force: "<<timer.getElapsedTime()<<std::endl;

    // update particle velocity
    unsigned int num_particle = this->pdm_base_->numSimParticles();
    for (unsigned int par_k = 0; par_k < num_particle; par_k++)
    {
        Vector<Scalar, Dim> vel_change = this->pdm_base_->particleForce(par_k)/this->pdm_base_->particleMass(par_k)*dt;
        this->pdm_base_->addParticleVelocity(par_k, vel_change);
        this->pdm_base_->setParticleVelocity(par_k, (1.0-this->vel_decay_ratio_)*this->pdm_base_->particleVelocity(par_k));
    }

    // boundary: set velocity for specified particle
    if (enable_boundary_condition_)
    {
        this->boundary_condition_method_->setSpecifiedParticleVel();
        this->boundary_condition_method_->resetSpecifiedParticleXYZVel();
        this->boundary_condition_method_->exertFloorBoundaryCondition();

        //need further consideration: only used for compression demo
        this->boundary_condition_method_->exertCompressionBoundaryCondition(dt);
    }

    // laplacian damping
    this->addLaplacianDamping();

    // update particle displacement
    for (unsigned int par_k = 0; par_k < num_particle; par_k++)
        this->pdm_base_->addParticleDisplacement(par_k, this->pdm_base_->particleVelocity(par_k)*dt);

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

    // boundary: reset fixed particle pos & vel
    if (enable_boundary_condition_)
    {
        this->boundary_condition_method_->resetSpecifiedParticlePos();
        this->boundary_condition_method_->resetSpecifiedParticleVel();
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
template class PDMStepMethodStateViscoPlasticitySemiImplicit<float,2>;
template class PDMStepMethodStateViscoPlasticitySemiImplicit<double,2>;
template class PDMStepMethodStateViscoPlasticitySemiImplicit<float,3>;
template class PDMStepMethodStateViscoPlasticitySemiImplicit<double,3>;

}//end of namespace Physika