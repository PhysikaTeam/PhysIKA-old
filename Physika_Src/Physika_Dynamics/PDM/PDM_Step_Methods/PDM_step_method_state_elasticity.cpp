/*
 * @file PDM_step_method_state.cpp 
 * @Basic PDMStepMethodstate class. basic step method class for state based PD, a simplest and straightforward explicit step method implemented
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

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_step_method_state_elasticity.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Fracture_Methods/PDM_fracture_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Topology_Control_Methods/PDM_topology_control_method_base.h"

#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Timer/timer.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMStepMethodStateElasticity<Scalar, Dim>::PDMStepMethodStateElasticity()
    :PDMStepMethodBase<Scalar,Dim>()
{

}

template <typename Scalar, int Dim>
PDMStepMethodStateElasticity<Scalar, Dim>::PDMStepMethodStateElasticity(PDMState<Scalar,Dim> * pdm_base)
    :PDMStepMethodBase<Scalar,Dim>(pdm_base)
{

}

template <typename Scalar, int Dim>
PDMStepMethodStateElasticity<Scalar,Dim>::~PDMStepMethodStateElasticity()
{

}


template <typename Scalar, int Dim>
void PDMStepMethodStateElasticity<Scalar,Dim>::calculateForce(Scalar dt)
{
    PDMState<Scalar,Dim> * pdm_state = dynamic_cast<PDMState<Scalar,Dim>*>(this->pdm_base_);
    if (pdm_state == NULL)
    {
        std::cerr<<"error in dynamic cast to PDMState *\n";
        std::exit(EXIT_FAILURE);
    }

    // for more detail of the following implementation, please refer to Erdogan et.al, 2014

    Scalar a;
    Scalar b;
    Scalar c;
    Scalar d;
    unsigned int num_particle = pdm_state->numSimParticles();

    Timer timer;
    timer.startTimer();

    for ( unsigned int par_k = 0; par_k <num_particle; par_k++)
    {       
        this->calculateParameters(pdm_state, par_k, a, b, c, d, DimensionTrait<Dim>());

        Scalar theta_k = this->calculateTheta(par_k, d);

        std::list<PDMFamily<Scalar, Dim> > & k_family = this->pdm_base_->particle(par_k).family();
        for (std::list<PDMFamily<Scalar, Dim> >::iterator j_iter = k_family.begin(); j_iter != k_family.end(); j_iter++)
        {
            // skip invalid family member
            if (j_iter->isVaild() == false || j_iter->isCrack() == true ) continue;

            unsigned int par_j = j_iter->id();  

            Scalar rest_rel_norm_kj = j_iter->restRelativePosNorm();
            Scalar cur_rel_norm_kj = j_iter->curRelativePosNorm();

            Scalar e_kj = cur_rel_norm_kj - rest_rel_norm_kj;
            Scalar s = e_kj/rest_rel_norm_kj;
            
            // fracture part
            if (this->enable_fracture_)
            {
                if (this->fracture_method_ == NULL)
                {
                    std::cerr<<"error: fracture_method not specified!\n";
                    std::exit(EXIT_FAILURE);
                }
                if (this->fracture_method_->applyFracture(s, par_k, k_family, j_iter) == true)
                {
                    //add crack element tuple
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

            Scalar delta = this->pdm_base_->delta(par_k);
            Scalar w_kj = delta/j_iter->weightRestLen();

            Scalar t_kj = 2*b*w_kj*e_kj;
            t_kj += 2*w_kj*d*a*theta_k*(j_iter->unitCurRelativePos().dot(j_iter->unitRestRelativePos()));
            Vector<Scalar,Dim> t_kj_v = (t_kj*particle_j.volume()*particle_k.volume())*j_iter->unitCurRelativePos();

            pdm_state->addParticleForce(par_k, t_kj_v);
            pdm_state->addParticleForce(par_j, -t_kj_v);

        }
    }

    timer.stopTimer();
    std::cout<<"time cost for internal force : "<<timer.getElapsedTime()<<std::endl;

}

template <typename Scalar, int Dim>
Scalar PDMStepMethodStateElasticity<Scalar, Dim>::calculateTheta(unsigned int par_k, Scalar d)
{
    // theta_k have to be calculated just only once
    Scalar theta_k = 0;
    Scalar delta = this->pdm_base_->delta(par_k);

    std::list<PDMFamily<Scalar, Dim> > & k_family = this->pdm_base_->particle(par_k).family();
    std::list<PDMFamily<Scalar, Dim> >::iterator k_family_end = k_family.end();   // to improve performance
    for (std::list<PDMFamily<Scalar, Dim> >::iterator l_iter = k_family.begin(); l_iter != k_family_end; l_iter++)
    {
        // skip invalid family member
        if (l_iter->isVaild() == false || l_iter->isCrack() == true ) continue;

        unsigned int par_l = l_iter->id();
        PDMParticle<Scalar,Dim> & particle_l = this->pdm_base_->particle(par_l);

        Scalar s_kl = l_iter->curRelativePosNorm()/l_iter->restRelativePosNorm() - 1;
        Scalar w_kl = delta/l_iter->weightRestLen();
        theta_k += w_kl*s_kl*particle_l.volume()*(l_iter->unitCurRelativePos().dot(l_iter->unitRestRelativePos())*l_iter->restRelativePosNorm());
    }
    theta_k = theta_k*d;
    return theta_k;
}

template <typename Scalar, int Dim>
void PDMStepMethodStateElasticity<Scalar,Dim>::calculateParameters(PDMState<Scalar,2> * pdm_state, unsigned int par_k, Scalar & a, Scalar & b, Scalar & c, Scalar & d, DimensionTrait<2> trait)
{
    Scalar delta = pdm_state->delta(par_k);
    Scalar delta_quad = delta*delta;
    delta_quad *= delta_quad;

    a = 0.5*(pdm_state->bulkModulus(par_k)-2.0*pdm_state->shearModulus(par_k));
    b = 6.0*pdm_state->shearModulus(par_k)/(PI*pdm_state->thickness()*delta_quad);
    c = 4.0*delta*b;
    d = 2.0/(PI*pdm_state->thickness()*delta_quad/delta);
}

template <typename Scalar, int Dim>
void PDMStepMethodStateElasticity<Scalar,Dim>::calculateParameters(PDMState<Scalar,3> * pdm_state, unsigned int par_k, Scalar & a, Scalar & b, Scalar & c, Scalar & d, DimensionTrait<3> trait)
{
    Scalar delta = pdm_state->delta(par_k);
    Scalar delta_quad = delta*delta;
    delta_quad *= delta_quad;

    a = 0.5*(pdm_state->bulkModulus(par_k) - 5.0/3.0*pdm_state->shearModulus(par_k));
    b = 15.0*pdm_state->shearModulus(par_k)/(2.0*PI*delta_quad*delta);
    c = 4.0*delta*b;
    d = 9.0/(4.0*PI*delta_quad);
}

// explicit instantiations
template class PDMStepMethodStateElasticity<float,2>;
template class PDMStepMethodStateElasticity<double,2>;
template class PDMStepMethodStateElasticity<float,3>;
template class PDMStepMethodStateElasticity<double,3>;

}// end of namespace Physika