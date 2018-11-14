/*
 * @file PDM_impact_method_sphere.cpp 
 * @brief  class of impact method based on Sphere collision detection for PDM drivers.
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

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Impact_Methods/PDM_impact_method_sphere.h"
#include "Physika_Dynamics/PDM/PDM_base.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMImpactMethodSphere<Scalar,Dim>::PDMImpactMethodSphere()
    :recovery_coefficient_(0.0), enable_project_pos_(true)
{

}

template <typename Scalar, int Dim>
PDMImpactMethodSphere<Scalar,Dim>::~PDMImpactMethodSphere()
{

}

template <typename Scalar, int Dim>
void PDMImpactMethodSphere<Scalar, Dim>::setRecoveryCoefficient(Scalar recovery_coefficient)
{
    this->recovery_coefficient_ = recovery_coefficient;
}

template <typename Scalar, int Dim>
void PDMImpactMethodSphere<Scalar, Dim>::enableProjectParticlePos()
{
    this->enable_project_pos_ = true;
}

template <typename Scalar, int Dim>
void PDMImpactMethodSphere<Scalar, Dim>::disableProjectParticlePos()
{
    this->enable_project_pos_ = false;
}

template <typename Scalar, int Dim>
void PDMImpactMethodSphere<Scalar,Dim>::applyImpact(Scalar dt)
{
    // trigger special treatment for critical time step
    this->triggerSpecialTreatment();

    PDMBase<Scalar,Dim> * driver = dynamic_cast<PDMBase<Scalar,Dim>*>(this->driver_);
    PHYSIKA_ASSERT(driver);

    // refresh impact pos
    this->impact_pos_ += dt*this->impact_velocity_;

    unsigned int numParticles = driver->numSimParticles();
    unsigned int contact_num = 0;

    for (long long par_idx = 0; par_idx < numParticles; par_idx++)
    {
        Vector<Scalar,Dim> pos = driver->particleCurrentPosition(par_idx);

        Vector<Scalar,Dim> relative_pos =  pos - impact_pos_;
        Scalar dist = relative_pos.norm();
        relative_pos.normalize();

        if (dist < impact_radius_)
        {
            contact_num ++;

            // reset position
            if (this->enable_project_pos_ == true)
            {
                Vector<Scalar, Dim> new_pos = impact_pos_ + impact_radius_* relative_pos;
                driver->setParticleCurPos(par_idx, new_pos);
            }
            
            // reset velocity
            Vector<Scalar, Dim> relative_vel = driver->particleVelocity(par_idx) - impact_velocity_;
            Scalar relative_vel_norm = relative_vel.norm();
            Scalar cos_theta = relative_vel.dot(relative_pos)/relative_vel_norm;
            if (cos_theta < 0)
            {
                Vector<Scalar, Dim> vel_normal = (relative_vel_norm*cos_theta)*relative_pos;
                Vector<Scalar, Dim> vel = impact_velocity_ - (1+recovery_coefficient_)*vel_normal + relative_vel;
                driver->setParticleVelocity(par_idx, vel);
            }                        
        }
    }

    std::cout<<"--------------------------------------------------------------------"<<std::endl;
    std::cout<<"impact pos: "<<this->impact_pos_<<std::endl;
    std::cout<<contact_num<<" particle is inside radius.\n";
    std::cout<<"--------------------------------------------------------------------"<<std::endl;
}

template <typename Scalar, int Dim>
void PDMImpactMethodSphere<Scalar, Dim>::triggerSpecialTreatment()
{
    if (this->trigger_special_treatment_ == true)
    {
        //need furthur consideration
        if (this->impact_pos_[1] - this->impact_radius_ <= 0.0)
        {
            //stop impact
            this->impact_velocity_ = Vector<Scalar, Dim>(0.0);

            std::cout<<"impact velocity: "<<this->impact_velocity_<<std::endl;
            std::cout<<"special treatment is triggered!"<<std::endl;
        }
    }
}

// explicit instantiations
template class PDMImpactMethodSphere<float,2>;
template class PDMImpactMethodSphere<double,2>;
template class PDMImpactMethodSphere<float,3>;
template class PDMImpactMethodSphere<double,3>;

}// end of namespace Physika