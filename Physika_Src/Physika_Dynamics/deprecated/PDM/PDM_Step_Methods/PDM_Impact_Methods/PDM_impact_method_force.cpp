/*
 * @file PDM_impact_method_force.cpp 
 * @brief  class of impact method based on projectile force for PDM drivers.
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

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Impact_Methods/PDM_impact_method_force.h"
#include "Physika_Dynamics/PDM/PDM_base.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMImpactMethodForce<Scalar,Dim>::PDMImpactMethodForce()
    :Ks_(0)
{

}

template <typename Scalar, int Dim>
PDMImpactMethodForce<Scalar,Dim>::~PDMImpactMethodForce()
{

}

template <typename Scalar, int Dim>
void PDMImpactMethodForce<Scalar,Dim>::setKs(Scalar Ks)
{
    this->Ks_ = Ks;
}

template <typename Scalar, int Dim>
void PDMImpactMethodForce<Scalar,Dim>::applyImpact(Scalar dt)
{
    PDMBase<Scalar,Dim> * driver = dynamic_cast<PDMBase<Scalar,Dim>*>(this->driver_);
    PHYSIKA_ASSERT(driver);

    //refresh impact pos
    this->impact_pos_ += dt*this->impact_velocity_;

    Vector<Scalar, Dim> vel_unit = impact_velocity_;
    vel_unit.normalize();
    unsigned int numParticles = driver->numSimParticles();
    unsigned int num = 0;
    for (unsigned int par_idx = 0; par_idx<numParticles; par_idx++)
    {
        Vector<Scalar,Dim> position = driver->particleCurrentPosition(par_idx);
        Scalar vol = driver->particle(par_idx).volume();

        Vector<Scalar,Dim> relative_pos =  position - impact_pos_;
        Scalar len = relative_pos.norm();
        if(len >= 0.75*impact_radius_ && len <= 1.5*impact_radius_)
        {
            num++;
            //std::cout<<"particle "<<par_idx << "is in impact radius.\n";
            Vector<Scalar,Dim> project_unit = relative_pos.normalize();
            Vector<Scalar,Dim> force_dir =(0.5*(vel_unit+project_unit));
            force_dir.normalize();

            Vector<Scalar,Dim> force = Ks_*pow((16*(len-0.75*impact_radius_)*pow(len-1.5*impact_radius_,2)),2.0/3)*vol*force_dir;
            driver->addParticleForce(par_idx,force);
        }
    }
    std::cout<<num<<" particle is inside radius.\n";
}

// explicit instantiations
template class PDMImpactMethodForce<float,2>;
template class PDMImpactMethodForce<float,3>;
template class PDMImpactMethodForce<double,2>;
template class PDMImpactMethodForce<double,3>;

}