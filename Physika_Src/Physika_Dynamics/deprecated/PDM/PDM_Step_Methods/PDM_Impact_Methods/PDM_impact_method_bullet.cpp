/*
 * @file PDM_impact_method_bullet.cpp 
 * @brief  class of impact method based on Bullet collision detection for PDM drivers.
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

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Impact_Methods/PDM_impact_method_bullet.h"
#include "Physika_Dynamics/PDM/PDM_base.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMImpactMethodBullet<Scalar,Dim>::PDMImpactMethodBullet()
    :bullet_cylinder_length_(0.0), recovery_coefficient_(0.0), enable_project_pos_(true)
{

}

template <typename Scalar, int Dim>
PDMImpactMethodBullet<Scalar,Dim>::~PDMImpactMethodBullet()
{

}

template <typename Scalar, int Dim>
void PDMImpactMethodBullet<Scalar, Dim>::setRecoveryCoefficient(Scalar recovery_coefficient)
{
    this->recovery_coefficient_ = recovery_coefficient;
}

template <typename Scalar, int Dim>
void PDMImpactMethodBullet<Scalar, Dim>::setBulletCylinderLength(Scalar bullet_cylinder_length)
{
    this->bullet_cylinder_length_ = bullet_cylinder_length;
}

template <typename Scalar, int Dim>
void PDMImpactMethodBullet<Scalar, Dim>::enableProjectParticlePos()
{
    this->enable_project_pos_ = true;
}

template <typename Scalar, int Dim>
void PDMImpactMethodBullet<Scalar, Dim>::disableProjectParticlePos()
{
    this->enable_project_pos_ = false;
}

template <typename Scalar, int Dim>
void PDMImpactMethodBullet<Scalar,Dim>::applyImpact(Scalar dt)
{
    PDMBase<Scalar,Dim> * driver = dynamic_cast<PDMBase<Scalar,Dim>*>(this->driver_);
    PHYSIKA_ASSERT(driver);

    // refresh impact pos
    this->impact_pos_ += dt*this->impact_velocity_;

    unsigned int numParticles = driver->numSimParticles();
    unsigned int sphere_contact_num = 0;
    unsigned int cylinder_contact_num = 0;

    Scalar impact_velocity_norm = this->impact_velocity_.norm();
    Vector<Scalar, Dim> unit_impact_velocity = this->impact_velocity_;
    unit_impact_velocity.normalize();

    for (long long par_idx = 0; par_idx < numParticles; par_idx++)
    {
        Vector<Scalar, Dim> pos = driver->particleCurrentPosition(par_idx);

        //for right sphere
        Vector<Scalar, Dim> relative_pos =  pos - impact_pos_;
        Vector<Scalar, Dim> unit_relative_pos = relative_pos;
        unit_relative_pos.normalize();
        Scalar sphere_dist = relative_pos.norm();
        bool is_inside_right_sphere = relative_pos.dot(unit_impact_velocity)>0 ? true:false;

        //for cylinder
        Scalar sin_theta = Vector<Scalar, Dim>(unit_relative_pos.cross(unit_impact_velocity)).norm(); // Vector<Scalar, Dim> is used to pass the complie
        if(Dim == 2) sin_theta /= 1.4142136;                                                          // special handling if Dim == 2

        Scalar cylinder_dist = sin_theta * sphere_dist;
        Vector<Scalar, Dim> relative_to_end_cylinder_pos = relative_pos + bullet_cylinder_length_*unit_impact_velocity;
        bool is_inside_cylinder = (is_inside_right_sphere == false) && (relative_to_end_cylinder_pos.dot(unit_impact_velocity) > 0);

        if (sphere_dist < impact_radius_ && is_inside_right_sphere)
        {
            sphere_contact_num ++;

            // reset position
            if (this->enable_project_pos_ == true)
            {
                Vector<Scalar, Dim> new_pos = impact_pos_ + impact_radius_* unit_relative_pos;
                driver->setParticleCurPos(par_idx, new_pos);
            }

            // reset velocity
            Vector<Scalar, Dim> relative_vel = driver->particleVelocity(par_idx) - impact_velocity_;
            Scalar relative_vel_norm = relative_vel.norm();
            Scalar cos_theta = relative_vel.dot(unit_relative_pos)/relative_vel_norm;
            if (cos_theta < 0)
            {
                Vector<Scalar, Dim> vel_normal = (relative_vel_norm*cos_theta)*unit_relative_pos;
                Vector<Scalar, Dim> new_vel = impact_velocity_ - (1+recovery_coefficient_)*vel_normal + relative_vel;
                driver->setParticleVelocity(par_idx, new_vel);
            }                        
        }
        else if (cylinder_dist < impact_radius_ && is_inside_cylinder)
        {
            cylinder_contact_num ++;

            Vector<Scalar, Dim> tangent_component_relative_pos = relative_pos.dot(unit_impact_velocity)*unit_impact_velocity;
            Vector<Scalar, Dim> normal_component_relative_pos = relative_pos - tangent_component_relative_pos;

            Vector<Scalar, Dim> unit_normal_component_relative_pos = normal_component_relative_pos;
            unit_normal_component_relative_pos.normalize();

            // reset position
            if (this->enable_project_pos_ == true)
            {
                Vector<Scalar, Dim> new_pos = pos + (impact_radius_ - cylinder_dist)*unit_normal_component_relative_pos;
                driver->setParticleCurPos(par_idx, new_pos);
            }

            //reset velocity
            Vector<Scalar, Dim> vel = driver->particleVelocity(par_idx);
            Scalar vel_normal_norm = vel.dot(unit_normal_component_relative_pos);
            if (vel_normal_norm < 0)
            {
                Vector<Scalar, Dim> vel_normal = vel_normal_norm*unit_normal_component_relative_pos;
                Vector<Scalar, Dim> new_vel = vel - (1+recovery_coefficient_)*vel_normal;
                driver->setParticleVelocity(par_idx, new_vel);
            }
        }
    }

    std::cout<<"--------------------------------------------------------------------"<<std::endl;
    std::cout<<"impact pos: "<<this->impact_pos_<<std::endl;
    std::cout<<"particle inside sphere: "<<sphere_contact_num<<std::endl;
    std::cout<<"particle is inside cylinder: "<<cylinder_contact_num<<std::endl;
    std::cout<<"--------------------------------------------------------------------"<<std::endl;
}

// explicit instantiations
template class PDMImpactMethodBullet<float,2>;
template class PDMImpactMethodBullet<double,2>;
template class PDMImpactMethodBullet<float,3>;
template class PDMImpactMethodBullet<double,3>;

}// end of namespace Physika