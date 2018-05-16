/*
 * @file PDM_step_method_bond.cpp 
 * @Basic PDMStepMethodBond class. basic step method class for bond based PD, a simplest and straightforward explicit step method implemented
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

#include "Physika_Dynamics/PDM/PDM_bond.h"
#include "Physika_Dynamics/PDM/PDM_bond_2d.h"

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_step_method_bond.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Fracture_Methods/PDM_fracture_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Impact_Methods/PDM_impact_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Topology_Control_Methods/PDM_topology_control_method_base.h"

#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Timer/timer.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMStepMethodBond<Scalar, Dim>::PDMStepMethodBond()
    :PDMStepMethodBase<Scalar,Dim>()
{

}

template <typename Scalar, int Dim>
PDMStepMethodBond<Scalar, Dim>::PDMStepMethodBond(PDMBond<Scalar,Dim> * pdm_base)
    :PDMStepMethodBase<Scalar,Dim>(pdm_base)
{
    
}

template <typename Scalar, int Dim>
PDMStepMethodBond<Scalar, Dim>::~PDMStepMethodBond()
{

}

template <typename Scalar, int Dim>
void PDMStepMethodBond<Scalar, Dim>::calculateForce(Scalar dt)
{
    PDMBond<Scalar, Dim> * pdm_bond = dynamic_cast<PDMBond<Scalar,Dim>*>(this->pdm_base_);
    if (pdm_bond == NULL)
    {
        std::cerr<<"error in dynamic cast to PDMBond *\n";
        std::exit(EXIT_FAILURE);
    }

    // for more detail of the following implementation, please refer to Erdogan et.al, 2014
    Scalar c;
    unsigned int num_particle = pdm_bond->numSimParticles();
   
    Timer timer;
    timer.startTimer();
    unsigned int fractrue_num = 0;
    for (unsigned int par_k = 0; par_k<num_particle; par_k++)
    {
        std::list<PDMFamily<Scalar,Dim> > & k_family = pdm_bond->particle(par_k).family();
        for (std::list<PDMFamily<Scalar,Dim> >::iterator j_iter = k_family.begin(); j_iter != k_family.end(); j_iter++)
        {
            // skip invalid family member
            if (j_iter->isVaild() == false || j_iter->isCrack() == true ) continue;


            this->calculateParameters(pdm_bond, par_k, c);
            unsigned int par_j = j_iter->id();

            // stretch s
            Scalar s = j_iter->curRelativePosNorm()/j_iter->restRelativePosNorm()-1.0;
     
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
                    fractrue_num++;

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

            // t_kj: the force of "per unit volume" of particle j exert on "per unit volume" of particle k
            // t_jk: the force of "per unit volume" of particle k exert on "per unit volume" of particle j
            PDMParticle<Scalar, Dim> & particle_j = pdm_bond->particle(par_j);
            PDMParticle<Scalar, Dim> & particle_k = pdm_bond->particle(par_k);

            Vector<Scalar, Dim> t_kj = (0.5*particle_j.volume() * particle_k.volume()*c*s)*j_iter->unitCurRelativePos();

            pdm_bond->addParticleForce(par_k, t_kj);
            pdm_bond->addParticleForce(par_j, -t_kj);
        }
    }
    timer.stopTimer();
    std::cout<<"time cost for internal force: "<<timer.getElapsedTime()<<std::endl;
    std::cout<<fractrue_num<<" reach critical s.\n";

}

template <typename Scalar, int Dim>
void PDMStepMethodBond<Scalar, Dim>::calculateParameters(PDMBond<Scalar,2> * pdm_bond, unsigned int par_k, Scalar & c)
{
    Scalar delta_cube = pdm_bond->delta(par_k)*pdm_bond->delta(par_k)*pdm_bond->delta(par_k);
    c = 12.0*pdm_bond->bulkModulus(par_k)/(PI*pdm_bond->thickness()*delta_cube);
}


template <typename Scalar, int Dim>
void PDMStepMethodBond<Scalar, Dim>::calculateParameters(PDMBond<Scalar,3> * pdm_bond, unsigned int par_k, Scalar & c)
{
    Scalar delta_quad = pdm_bond->delta(par_k)*pdm_bond->delta(par_k);
    delta_quad *= delta_quad;
    c = 18.0*pdm_bond->bulkModulus()/(PI*delta_quad);
}


// explicit instantiations
template class PDMStepMethodBond<float, 2>;
template class PDMStepMethodBond<double,2>;
template class PDMStepMethodBond<float, 3>;
template class PDMStepMethodBond<double,3>;

}// end of namespace Physika