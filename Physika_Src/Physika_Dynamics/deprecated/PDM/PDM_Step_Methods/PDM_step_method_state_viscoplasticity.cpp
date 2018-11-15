/*
 * @file PDM_step_method_state_viscoplasticiy.cpp
 * @Basic PDMStepMethodStateViscoPlasticity class.
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

#include <iostream>
#include <fstream>

#include "Physika_Dynamics/PDM/PDM_state.h"
#include "Physika_Dynamics/PDM/PDM_state_2d.h"

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Fracture_Methods/PDM_fracture_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Impact_Methods/PDM_impact_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Topology_Control_Methods/PDM_topology_control_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Boundary_Condition_Methods/PDM_boundary_condition_method.h"

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_step_method_state_viscoplasticity.h"

#include "Physika_Core/Timer/timer.h"
#include "Physika_Core/Utilities/math_utilities.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMStepMethodStateViscoPlasticity<Scalar, Dim>::PDMStepMethodStateViscoPlasticity()
    :relax_time_(0.001), lambda_(1.0), Kd_(0.0), laplacian_damping_coefficient_(0.0), laplacian_damping_iter_times_(0),
    vel_decay_ratio_(0.0), Rcp_(1.0), Ecp_limit_(1.0), enable_plastic_statistics(false)
{

}

template <typename Scalar, int Dim>
PDMStepMethodStateViscoPlasticity<Scalar, Dim>::PDMStepMethodStateViscoPlasticity(PDMState<Scalar, Dim> * pdm_base)
    :PDMStepMethodStateElasticity(pdm_base),relax_time_(0.001), lambda_(1.0), Kd_(0.0), 
    laplacian_damping_coefficient_(0.0), laplacian_damping_iter_times_(0),
    vel_decay_ratio_(0.0), Rcp_(1.0), Ecp_limit_(1.0), enable_plastic_statistics(false)
{
    this->setPDMDriver(pdm_base);
}

template <typename Scalar, int Dim>
PDMStepMethodStateViscoPlasticity<Scalar, Dim>::~PDMStepMethodStateViscoPlasticity()
{

}

template <typename Scalar, int Dim>
void PDMStepMethodStateViscoPlasticity<Scalar, Dim>::setPDMDriver(PDMBase<Scalar, Dim> * pdm_base)
{
    this->PDMStepMethodStateElasticity<Scalar, Dim>::setPDMDriver(pdm_base);

    this->yield_critical_val_vec_.clear();
    for (unsigned int par_id = 0; par_id < pdm_base->numSimParticles(); par_id++)
        this->yield_critical_val_vec_.push_back(std::numeric_limits<Scalar>::max());
}

template <typename Scalar, int Dim>
void PDMStepMethodStateViscoPlasticity<Scalar, Dim>::setKd(Scalar Kd)
{
    this->Kd_ = Kd;
}

template <typename Scalar, int Dim>
void PDMStepMethodStateViscoPlasticity<Scalar, Dim>::setVelDecayRatio(Scalar vel_decay_ratio)
{
    this->vel_decay_ratio_ = vel_decay_ratio;
}

template <typename Scalar, int Dim>
void PDMStepMethodStateViscoPlasticity<Scalar, Dim>::setLaplacianDampingCoefficient(Scalar laplacian_damping_coefficient)
{
    this->laplacian_damping_coefficient_ = laplacian_damping_coefficient;
}

template <typename Scalar, int Dim>
void PDMStepMethodStateViscoPlasticity<Scalar, Dim>::setLaplacianDampingIterTimes(unsigned int laplacian_damping_iter_times)
{
    this->laplacian_damping_iter_times_ = laplacian_damping_iter_times;
}

template <typename Scalar, int Dim>
void PDMStepMethodStateViscoPlasticity<Scalar, Dim>::setRcp(Scalar Rcp)
{
    this->Rcp_ = Rcp;
}

template <typename Scalar, int Dim>
void PDMStepMethodStateViscoPlasticity<Scalar, Dim>::setEcpLimit(Scalar Ecp_limit)
{
    this->Ecp_limit_ = Ecp_limit;
}

template <typename Scalar, int Dim>
void PDMStepMethodStateViscoPlasticity<Scalar, Dim>::setRelaxTime(Scalar relax_time)
{
    this->relax_time_ = relax_time;
}

template <typename Scalar, int Dim>
void PDMStepMethodStateViscoPlasticity<Scalar, Dim>::setLambda(Scalar lambda)
{
    this->lambda_ = lambda;
}

template <typename Scalar, int Dim>
void PDMStepMethodStateViscoPlasticity<Scalar, Dim>::setHomogeneousYieldCriticalVal(Scalar yield_critical_val)
{
    PHYSIKA_ASSERT(this->yield_critical_val_vec_.size() > 0);

    for (unsigned int par_id = 0; par_id < this->yield_critical_val_vec_.size(); par_id++)
        this->yield_critical_val_vec_[par_id] = yield_critical_val;
}

template <typename Scalar, int Dim>
void PDMStepMethodStateViscoPlasticity<Scalar, Dim>::setYieldCriticalVal(unsigned int par_id , Scalar yield_critical_val)
{
    PHYSIKA_ASSERT(this->yield_critical_val_vec_.size() > 0);

    this->yield_critical_val_vec_[par_id] = yield_critical_val;

}

template <typename Scalar, int Dim>
Scalar PDMStepMethodStateViscoPlasticity<Scalar, Dim>::yieldCriticalVal(unsigned int par_id) const
{
    return this->yield_critical_val_vec_[par_id];
}

template <typename Scalar, int Dim>
void PDMStepMethodStateViscoPlasticity<Scalar, Dim>::enablePlasticStatistics()
{
    this->enable_plastic_statistics = true;
}

template <typename Scalar, int Dim>
void PDMStepMethodStateViscoPlasticity<Scalar, Dim>::disablePlasticStatistics()
{
    this->enable_plastic_statistics = false;
}

template <typename Scalar, int Dim>
bool PDMStepMethodStateViscoPlasticity<Scalar, Dim>::isPlasticStatisticsEnabled() const
{
    return this->enable_plastic_statistics;
}

template <typename Scalar, int Dim>
void PDMStepMethodStateViscoPlasticity<Scalar, Dim>::calculateForce(Scalar dt)
{
    PDMState<Scalar,Dim> * pdm_state = dynamic_cast<PDMState<Scalar,Dim>*>(this->pdm_base_);
    if (pdm_state == NULL)
    {
        std::cerr<<"error in dynamic cast to PDMState *\n";
        std::exit(EXIT_FAILURE);
    }

    Scalar a_1; // a_1 = 0.5*k
    Scalar a_2; // a_2 = -5.0/6.0*u, and a_1 + a_2 = a
    Scalar b;
    Scalar c;
    Scalar d;
    unsigned int num_particle = pdm_state->numSimParticles();

    //----------------------------------------------------------------------------
    unsigned int fracture_s_num = 0;
    unsigned int fracture_p_num = 0;

    unsigned int global_family_num =   0;
    Scalar global_s_max =              0;
    Scalar global_e_minus_ep_max =     0;
    Scalar global_cur_e_max =          0;
    Scalar global_cur_eb_max =         0;
    Scalar global_cur_ep_max =         0;
    Scalar global_yield_function_max = 0;
    Scalar total_e = 0;
    Scalar total_eb = 0;
    Scalar total_ep = 0;
    //-----------------------------------------------------------------------------

    //#pragma omp parallel for private(a_1, a_2, b, c, d)
    for (long long par_k = 0; par_k < num_particle; par_k++)
    {
        this->calculateParameters(pdm_state, par_k, a_1, a_2, b, c, d, DimensionTrait<Dim>());

        // calculate theta_k
        Scalar theta_k = this->calculateTheta(par_k, d);

        std::list<PDMFamily<Scalar, Dim> > & k_family = this->pdm_base_->particle(par_k).family();
        std::list<PDMFamily<Scalar, Dim> >::iterator k_family_end = k_family.end();  
        for (std::list<PDMFamily<Scalar, Dim> >::iterator j_iter = k_family.begin(); j_iter != k_family.end(); j_iter++)
        {
            // skip invalid family member
            if (j_iter->isVaild() == false || j_iter->isCrack() == true ) continue;

            const Vector<Scalar, Dim> & rest_unit_rel_kj = j_iter->unitRestRelativePos();
            const Vector<Scalar, Dim> & cur_unit_rel_kj = j_iter->unitCurRelativePos();

            Scalar rest_rel_norm_kj = j_iter->restRelativePosNorm();
            Scalar cur_rel_norm_kj = j_iter->curRelativePosNorm();

            
            Scalar cur_e = cur_rel_norm_kj - rest_rel_norm_kj;
            Scalar pre_eb = j_iter->eb();
            Scalar cur_eb = pre_eb;
            Scalar pre_ep = j_iter->ep();
            Scalar cur_ep = pre_ep;
            
            Scalar delta = this->pdm_base_->delta(par_k);
            Scalar w_kj = delta/j_iter->weightRestLen();

            //update last ed which is necessary for next step
            Scalar last_ed = j_iter->ed();
            Scalar cur_ed =cur_e + d*a_2*theta_k*(cur_unit_rel_kj.dot(rest_unit_rel_kj))/b;
            j_iter->setEd(cur_ed);  

            //----------------------------------------------------------------------------
            //ViscoElasticity
            //----------------------------------------------------------------------------
            Scalar relax_time = this->relax_time_;
            Scalar e_pow = std::exp(-dt/relax_time);
            Scalar e_mediate = 1.0 - e_pow;
            Scalar beta = 1.0 - relax_time/dt*e_mediate;

            
            Scalar delta_ed = cur_ed - last_ed;
            Scalar delta_eb = (last_ed-pre_ep-pre_eb)*e_mediate + beta*delta_ed;

            cur_eb += delta_eb;

            /*
            if (delta_eb < 0.0)
                cur_eb += this->plastic_compress_ratio_*delta_eb;
            else
                cur_eb += delta_eb;
            */

            //update eb
            cur_eb = cur_eb*min(static_cast<Scalar>(1.0), j_iter->ebLimit()/abs(cur_eb));
            j_iter->setEb(cur_eb);      

            //----------------------------------------------------------------------------
            //Plasticity
            //----------------------------------------------------------------------------
            Scalar t_d_trial = 2.0*w_kj*b*(cur_ed - pre_ep - this->lambda_*cur_eb);
            Scalar t_d = t_d_trial;
            Scalar yield_critical_value = this->yieldCriticalVal(par_k);

            //need further consideration
            if (cur_ed < pre_ep) yield_critical_value *= this->Rcp_;

            Scalar yield_function_value =  0.5*t_d_trial*t_d_trial;

            if (yield_function_value > yield_critical_value ) // if material is under plastic flow
            {
                t_d = sqrt(2.0*yield_critical_value)*(t_d_trial>0?1:-1);

                //update ep
                Scalar delta_lambda = (abs(t_d_trial)/sqrt(2.0*yield_critical_value)-1.0)/(2.0*b*w_kj);
                Scalar delta_ep = delta_lambda*t_d;

                cur_ep += delta_ep;
                if (cur_ep < 0)
                    cur_ep = cur_ep*min(static_cast<Scalar>(1.0), j_iter->epLimit()/(abs(cur_ep)*this->Ecp_limit_));
                else
                    cur_ep = cur_ep*min(static_cast<Scalar>(1.0), j_iter->epLimit()/abs(cur_ep));

                j_iter->setEp(cur_ep);

                //if ep had no change, reassign to t_d_trial
                if (isEqual(pre_ep, cur_ep) == true) t_d = t_d_trial;
            }

            //---------------------------------------------------------------------------------------------
            if (enable_plastic_statistics)
            {
                #pragma omp atomic
                global_family_num++;
                #pragma omp atomic
                total_eb += abs(cur_eb);
                #pragma omp atomic
                total_e += abs(cur_e);
                #pragma omp atomic
                total_ep += abs(cur_ep);
                if (abs(global_cur_e_max) < abs(cur_e)) global_cur_e_max = cur_e;
                if (abs(global_cur_ep_max) < abs(cur_ep)) global_cur_ep_max = cur_ep;
                if (abs(global_cur_eb_max) < abs(cur_eb)) global_cur_eb_max = cur_eb;
                if (global_yield_function_max < yield_function_value) global_yield_function_max = yield_function_value;
            }
            //---------------------------------------------------------------------------------------------

            Scalar e_minus_ep = cur_e - j_iter->ep();
            Scalar s = e_minus_ep/rest_rel_norm_kj;

            if (abs(global_s_max) < abs(s)) global_s_max = s;
            if (abs(global_e_minus_ep_max) < abs(e_minus_ep)) global_e_minus_ep_max = e_minus_ep;

            unsigned int par_j = j_iter->id();

            // fracture and topology control part
            if (this->enable_fracture_)
            {
                if (this->fracture_method_ == NULL)
                {
                    std::cerr<<"error: fracture_method not specified!\n";
                    std::exit(EXIT_FAILURE);
                }

                if (this->fracture_method_->applyFracture(cur_e-j_iter->ep(), par_k, k_family, j_iter) == true)
                {
                    fracture_s_num++;

                    if (this->enable_topology_control_)
                    {
                        if (this->topology_control_method_ == NULL)
                        {
                            std::cerr<<"error: topology_control_method not specified!\n";
                            std::exit(EXIT_FAILURE);
                        }

                        #pragma omp critical (TOPOLOGY_CONTROL)
                        this->topology_control_method_->addElementTuple(par_k, par_j); //note: par_j != j_iter->id()
                    }

                    continue;
                }
            }

            PDMParticle<Scalar, Dim> & particle_j = pdm_state->particle(par_j);
            PDMParticle<Scalar, Dim> & particle_k = pdm_state->particle(par_k);

            
            Scalar t_i = (2.0*w_kj*d*a_1*theta_k)*(cur_unit_rel_kj.dot(rest_unit_rel_kj));
            Scalar t_kj = t_i + t_d;
            Vector<Scalar,Dim> t_kj_v = (t_kj*particle_j.volume()*particle_k.volume())*cur_unit_rel_kj;

            // omp atomic for Vector<Scalar,3> is used to prevent collision
            pdm_state->addParticleForce(par_k, t_kj_v);
            pdm_state->addParticleForce(par_j, -t_kj_v);
        }

    }

    std::cout<<"--------------------------------------------------------------------"<<std::endl;
    std::cout<<"fracture by critical s: "<<fracture_s_num<<std::endl;
    std::cout<<"fracture by plasticity: "<<fracture_p_num<<std::endl;
    std::cout<<"--------------------------statistics--------------------------"<<std::endl;
    std::cout<<"global s_max:                  "<<global_s_max<<std::endl;
    std::cout<<"global e-ep max:               "<<global_e_minus_ep_max<<std::endl;
    std::cout<<"global cur_e_max:              "<<global_cur_e_max<<std::endl;
    std::cout<<"global cur_eb_max:             "<<global_cur_eb_max<<std::endl;
    std::cout<<"global cur_ep_max:             "<<global_cur_ep_max<<std::endl;
    std::cout<<"global yield_function_max:     "<<global_yield_function_max<<std::endl;
    std::cout<<"global family num:             "<<global_family_num<<std::endl;
    std::cout<<"total e:                       "<<total_e<<std::endl;
    std::cout<<"total eb:                      "<<total_eb<<std::endl;
    std::cout<<"total ep:                      "<<total_ep<<std::endl;
    std::cout<<"global ratio eb/e:             "<<total_eb/total_e<<std::endl;
    std::cout<<"global ratio ep/e:             "<<total_ep/total_e<<std::endl;
    std::cout<<"--------------------------------------------------------------------"<<std::endl;
}

template <typename Scalar, int Dim>
void PDMStepMethodStateViscoPlasticity<Scalar, Dim>::addDampingForce()
{
    //need further consideration
    /////////////////////////////////////////////////////////////////////////////////////////
    std::fstream file("Kd.txt", std::ios::in);
    if (file.fail() == false)
        file>>this->Kd_;
    file.close();
    std::cout<<"Kd: "<<this->Kd_<<std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////

    PDMState<Scalar,Dim> * pdm_state = dynamic_cast<PDMState<Scalar,Dim>*>(this->pdm_base_);
    if (pdm_state == NULL)
    {
        std::cerr<<"error in dynamic cast to PDMState *\n";
        std::exit(EXIT_FAILURE);
    }

    for (long long par_k = 0; par_k < pdm_state->numSimParticles(); par_k++)
    {
        PDMParticle<Scalar, Dim> & particle_k = pdm_state->particle(par_k);
        Vector<Scalar, Dim> damping_force = (-this->Kd_*particle_k.volume())*this->pdm_base_->particleVelocity(par_k);

        //need further consideration
        //if (particle_k.validFamilySize() < 10) damping_force /= 5.0;
            
        pdm_state->addParticleForce(par_k, damping_force);
    }
}

template <typename Scalar, int Dim>
void PDMStepMethodStateViscoPlasticity<Scalar, Dim>::addLaplacianDamping()
{
    if (this->laplacian_damping_iter_times_ == 0) return;

    //need further consideration
    /////////////////////////////////////////////////////////////////////////////////////////
    std::fstream file("laplacian_damping.txt", std::ios::in);
    if (file.fail() == false)
    {
        file>>this->laplacian_damping_coefficient_;
        file>>this->laplacian_damping_iter_times_;
        std::cout<<"laplacian_damping_coefficient: "<<this->laplacian_damping_coefficient_<<std::endl;
        std::cout<<"laplacian_damping_iter_times:  "<<this->laplacian_damping_iter_times_<<std::endl;
    }
    file.close();
    ////////////////////////////////////////////////////////////////////////////////////////

    //define new_vel_vec
    std::vector<Vector<Scalar, Dim> > new_vel_vec(this->pdm_base_->numSimParticles());

    #pragma omp parallel for
    for (long long par_k = 0; par_k < this->pdm_base_->numSimParticles(); par_k++)
        new_vel_vec[par_k] = this->pdm_base_->particleVelocity(par_k);

    //laplacian damping
    for (unsigned int iter_id = 0; iter_id < this->laplacian_damping_iter_times_; iter_id++)
    {
        //add damping
        #pragma omp parallel for
        for (long long par_k = 0; par_k < this->pdm_base_->numSimParticles(); par_k++)
        {
            std::list<PDMFamily<Scalar, Dim> > & k_family = this->pdm_base_->particle(par_k).family();
            std::list<PDMFamily<Scalar, Dim> >::iterator k_family_end = k_family.end();  
            for (std::list<PDMFamily<Scalar, Dim> >::iterator j_iter = k_family.begin(); j_iter != k_family.end(); j_iter++)
            {
                // skip invalid family member
                if (j_iter->isVaild() == false || j_iter->isCrack() == true ) continue;

                const Vector<Scalar, Dim> & par_k_vel = this->pdm_base_->particleVelocity(par_k);
                const Vector<Scalar, Dim> & par_j_vel = this->pdm_base_->particleVelocity(j_iter->id());

                new_vel_vec[par_k] += this->laplacian_damping_coefficient_*(par_j_vel - par_k_vel);
            }
        }

        //set new particle velocity
        #pragma omp parallel for
        for (long long par_k = 0; par_k < this->pdm_base_->numSimParticles(); par_k++)
            this->pdm_base_->setParticleVelocity(par_k, new_vel_vec[par_k]);        
    }
}

template <typename Scalar, int Dim>
void PDMStepMethodStateViscoPlasticity<Scalar, Dim>::advanceStep(Scalar dt)
{
    Timer timer;

    // update family parameters
    timer.startTimer();
    this->updateParticleFamilyParameters();
    timer.stopTimer();
    std::cout<<"time for update family parameters: "<<timer.getElapsedTime()<<std::endl;

    // reset force
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

    // calculate Force
    this->calculateForce(dt);

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
}


template <typename Scalar, int Dim>
void PDMStepMethodStateViscoPlasticity<Scalar,Dim>::calculateParameters(PDMState<Scalar,2> * pdm_state, unsigned int par_k, Scalar &a_1, Scalar & a_2, Scalar & b, Scalar & c, Scalar & d, DimensionTrait<2> trait)
{
    Scalar delta = pdm_state->delta(par_k);
    Scalar delta_quad = delta*delta;
    delta_quad *= delta_quad;

    a_1 = 0.5*pdm_state->bulkModulus(par_k);
    a_2 = -pdm_state->shearModulus(par_k);
    b = 6.0*pdm_state->shearModulus(par_k)/(PI*pdm_state->thickness()*delta_quad);
    c = 4.0*delta*b;
    d = 2.0/(PI*pdm_state->thickness()*delta_quad/delta);
}

template <typename Scalar, int Dim>
void PDMStepMethodStateViscoPlasticity<Scalar,Dim>::calculateParameters(PDMState<Scalar,3> * pdm_state, unsigned int par_k, Scalar &a_1, Scalar & a_2, Scalar & b, Scalar & c, Scalar & d, DimensionTrait<3> trait)
{
    Scalar delta = pdm_state->delta(par_k);
    Scalar delta_quad = delta*delta;
    delta_quad *= delta_quad;

    a_1 = 0.5*pdm_state->bulkModulus(par_k);
    a_2 = -5.0/6.0*pdm_state->shearModulus(par_k);
    b = 15.0*pdm_state->shearModulus(par_k)/(2.0*PI*delta_quad*delta);
    c = 4.0*delta*b;
    d = 9.0/(4.0*PI*delta_quad);
}

//explicit instantiations
template class PDMStepMethodStateViscoPlasticity<float,2>;
template class PDMStepMethodStateViscoPlasticity<double,2>;
template class PDMStepMethodStateViscoPlasticity<float,3>;
template class PDMStepMethodStateViscoPlasticity<double,3>;

}//end of namespace Physika