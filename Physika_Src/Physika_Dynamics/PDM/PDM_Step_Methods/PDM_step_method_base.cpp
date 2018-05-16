/*
 * @file PDM_step_method_base.cpp
 * @Basic PDMStepMethodBase class. basic step method class, a simplest and straightforward explicit step method implemented
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

#include "Physika_Dynamics/PDM/PDM_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_step_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Fracture_Methods/PDM_fracture_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Impact_Methods/PDM_impact_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Topology_Control_Methods/PDM_topology_control_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Boundary_Condition_Methods/PDM_boundary_condition_method.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMStepMethodBase<Scalar, Dim>::PDMStepMethodBase()
    :enable_fracture_(false),fracture_method_(NULL),enable_collision_(false),collision_method_(NULL),
    enable_impact_(false),impact_method_(NULL),
    enable_topology_control_(false), topology_control_method_(NULL), enable_boundary_condition_(false), boundary_condition_method_(NULL)
{

}

template <typename Scalar, int Dim>
PDMStepMethodBase<Scalar, Dim>::PDMStepMethodBase(PDMBase<Scalar,Dim> * pdm_base)
    :enable_fracture_(false),fracture_method_(NULL),enable_collision_(false),collision_method_(NULL),
    enable_impact_(false),impact_method_(NULL),
    enable_topology_control_(false), topology_control_method_(NULL), enable_boundary_condition_(false), boundary_condition_method_(NULL)
{
    this->setPDMDriver(pdm_base);
}

template <typename Scalar, int Dim>
PDMStepMethodBase<Scalar, Dim>::~PDMStepMethodBase()
{

}

template <typename Scalar, int Dim>
PDMBase<Scalar, Dim> * PDMStepMethodBase<Scalar, Dim>::PDMDriver()
{
    return this->pdm_base_;
}

template <typename Scalar, int Dim>
const PDMBase<Scalar, Dim> * PDMStepMethodBase<Scalar, Dim>::PDMDriver() const
{
    return this->pdm_base_;
}

template <typename Scalar, int Dim>
void PDMStepMethodBase<Scalar, Dim>::setPDMDriver(PDMBase<Scalar, Dim> * pdm_base)
{
    if (pdm_base == NULL)
    {
        std::cerr<<"Error: Cannot set NULL PDM driver to PDMStepMethodBase, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    this->pdm_base_ = pdm_base;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Scalar, int Dim>
void PDMStepMethodBase<Scalar, Dim>::setFractureMethod(PDMFractureMethodBase<Scalar,Dim> * fracture_method)
{
    if (fracture_method == NULL)
    {
        std::cerr<<"Error: Cannot set NULL Fracture Method to PDMStepMethodBase, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    this->fracture_method_ = fracture_method;
    this->fracture_method_->setDriver(this->pdm_base_);
}

template <typename Scalar, int Dim>
void PDMStepMethodBase<Scalar,Dim>::enableFracture()
{
    this->enable_fracture_ = true;
}

template <typename Scalar, int Dim>
void PDMStepMethodBase<Scalar,Dim>::disableFracture()
{
    this->enable_fracture_ = false;
}

template <typename Scalar, int Dim>
bool PDMStepMethodBase<Scalar,Dim>::isFractureEnabled() const
{
    return this->enable_fracture_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Scalar, int Dim>
void PDMStepMethodBase<Scalar, Dim>::setCollisionMethod(PDMCollisionMethodBase<Scalar,Dim> * collision_method)
{
    if (collision_method == NULL)
    {
        std::cerr<<"Error: Cannot set NULL Collision Method to PDMStepMethodBase, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    this->collision_method_ = collision_method;
    this->collision_method_->setDriver(this->pdm_base_);
}

template <typename Scalar, int Dim>
void PDMStepMethodBase<Scalar,Dim>::enableCollision()
{
    this->enable_collision_ = true;
}

template <typename Scalar, int Dim>
void PDMStepMethodBase<Scalar,Dim>::disableCollision()
{
    this->enable_collision_ = false;
}

template <typename Scalar, int Dim>
bool PDMStepMethodBase<Scalar,Dim>::isCollisionEnabled() const
{
    return this->enable_collision_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Scalar, int Dim>
void PDMStepMethodBase<Scalar, Dim>::setImpactMethod(PDMImpactMethodBase<Scalar,Dim> * impact_method)
{
    if (impact_method == NULL)
    {
        std::cerr<<"Error: Cannot set NULL Impact Method to PDMStepMethodBase, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    this->impact_method_ = impact_method;
    this->impact_method_->setDriver(this->pdm_base_);
}

template <typename Scalar, int Dim>
void PDMStepMethodBase<Scalar,Dim>::enableImpact()
{
    this->enable_impact_ = true;
}

template <typename Scalar, int Dim>
void PDMStepMethodBase<Scalar,Dim>::disableImpact()
{
    this->enable_impact_ = false;
}

template <typename Scalar, int Dim>
bool PDMStepMethodBase<Scalar,Dim>::isImpactEnabled() const
{
    return this->enable_impact_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Scalar, int Dim>
void PDMStepMethodBase<Scalar, Dim>::setTopologyControlMethod(PDMTopologyControlMethodBase<Scalar,Dim> * topology_control_method)
{
    if (topology_control_method == NULL)
    {
        std::cerr<<"Error: Cannot set NULL Topology Control Method to PDMStepMethodBase, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    this->topology_control_method_ = topology_control_method;
    this->topology_control_method_->setDriver(this->pdm_base_);
}

template <typename Scalar, int Dim>
void PDMStepMethodBase<Scalar,Dim>::enableTopologyControl()
{
    this->enable_topology_control_ = true;
}

template <typename Scalar, int Dim>
void PDMStepMethodBase<Scalar,Dim>::disableTopologyControl()
{
    this->enable_topology_control_ = false;
}

template <typename Scalar, int Dim>
bool PDMStepMethodBase<Scalar,Dim>::isTopologyControlEnabled() const
{
    return this->enable_topology_control_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Scalar, int Dim>
void PDMStepMethodBase<Scalar, Dim>::setBoundaryConditionMethod(PDMBoundaryConditionMethod<Scalar, Dim> * boundary_condition_method)
{
    if (boundary_condition_method == NULL)
    {
        std::cerr<<"Error: Cannot set NULL Topology Control Method to PDMStepMethodBase, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    this->boundary_condition_method_ = boundary_condition_method;
    this->boundary_condition_method_->setDriver(this->pdm_base_);
}

template <typename Scalar, int Dim>
void PDMStepMethodBase<Scalar,Dim>::enableBoundaryCondition()
{
    this->enable_boundary_condition_ = true;
}

template <typename Scalar, int Dim>
void PDMStepMethodBase<Scalar,Dim>::disableBoundaryCondition()
{
    this->enable_boundary_condition_ = false;
}

template <typename Scalar, int Dim>
bool PDMStepMethodBase<Scalar,Dim>::isBoundaryConditionEnabled() const
{
    return this->enable_boundary_condition_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Scalar, int Dim>
void PDMStepMethodBase<Scalar, Dim>::advanceStep(Scalar dt)
{
    Timer timer;

    // update family parameters
    timer.startTimer();
    this->updateParticleFamilyParameters();
    timer.stopTimer();
    std::cout<<"time for update family parameters: "<<timer.getElapsedTime()<<std::endl;

    //reset force
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

    // update particle velocity
    unsigned int num_particle = this->pdm_base_->numSimParticles();
    for (unsigned int par_k = 0; par_k < num_particle; par_k++)
    {
        Vector<Scalar, Dim> vel_change = this->pdm_base_->particleForce(par_k)/this->pdm_base_->particleMass(par_k)*dt;
        this->pdm_base_->addParticleVelocity(par_k, vel_change);
    }

    // boundary: set velocity for specified particle
    if (enable_boundary_condition_)
    {
        this->boundary_condition_method_->setSpecifiedParticleVel();
        this->boundary_condition_method_->exertFloorBoundaryCondition();
    }

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

template <typename Scalar, int Dim>
void PDMStepMethodBase<Scalar, Dim>::updateParticleFamilyParameters()
{
    unsigned int num_particle = this->pdm_base_->numSimParticles();
    Vector<Scalar, Dim> cur_relative_pos(0.0);

    #pragma omp parallel for firstprivate(cur_relative_pos)
    for (long long par_k = 0; par_k<num_particle; par_k++)
    {
        std::list<PDMFamily<Scalar, Dim> > & k_family = this->pdm_base_->particle(par_k).family();
        std::list<PDMFamily<Scalar, Dim> >::iterator k_family_end = k_family.end();   // to improve performance
        for (std::list<PDMFamily<Scalar, Dim> >::iterator j_iter = k_family.begin(); j_iter!=k_family_end; j_iter++)
        {
            unsigned int par_j = j_iter->id();
            cur_relative_pos = this->pdm_base_->particleCurrentPosition(par_j)-this->pdm_base_->particleCurrentPosition(par_k);
            j_iter->setCurRelativePos(cur_relative_pos);
        }
    }
}

template <typename Scalar, int Dim>
void PDMStepMethodBase<Scalar, Dim>::updateSpecifiedParticleFamilyParameters(unsigned int par_k, PDMFamily<Scalar, Dim> & pdm_family)
{
    unsigned int par_j = pdm_family.id();
    Vector<Scalar, Dim> cur_relative_pos = this->pdm_base_->particleCurrentPosition(par_j)-this->pdm_base_->particleCurrentPosition(par_k);
    pdm_family.setCurRelativePos(cur_relative_pos);
}

template <typename Scalar, int Dim>
void PDMStepMethodBase<Scalar, Dim>::addGravity()
{
    unsigned int num_particle = pdm_base_->numSimParticles();
    for (unsigned int par_k = 0; par_k < num_particle; par_k ++)
    {
        PDMParticle<Scalar, Dim> & particle_k = pdm_base_->particle(par_k);
        Vector<Scalar, Dim> gravity_force(0);
        gravity_force[1] = -1.0*pdm_base_->gravity();
        gravity_force *= particle_k.mass();
        pdm_base_->addParticleForce(par_k, gravity_force);
    }
}

//explicit instantiations
template class PDMStepMethodBase<float,2>;
template class PDMStepMethodBase<double,2>;
template class PDMStepMethodBase<float,3>;
template class PDMStepMethodBase<double,3>;

}// end of namespace Physika