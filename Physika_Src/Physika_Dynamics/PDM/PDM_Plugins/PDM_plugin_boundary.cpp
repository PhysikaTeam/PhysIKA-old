/*
 * @file PDM_plugin_render.h 
 * @brief boundary plugins class for PDM drivers. It will exert boundary conditions to PDM systems
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

#include <cmath>
#include "Physika_Dynamics/PDM/PDM_Plugins/PDM_plugin_boundary.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
namespace Physika{

template <typename Scalar, int Dim>
PDMPluginBoundary<Scalar,Dim>::PDMPluginBoundary()
    :PDMPluginBase(),enable_floor_(false),enable_force_(false),enable_fix_(false),enable_eliminate_offset_(false),enable_bullet_(false),enable_sphere_(false),
    floor_pos_(0.0),recovery_coefficient_(0.99),eliminate_x_offset_(true),eliminate_y_offset_(false),eliminate_z_offset_(true),bullet_pos_(0),bullet_velocity_(0),
    bullet_radius_(0.0),Ks_(0.0),sphere_radius_(0.0),sphere_pos_(0.0),sphere_velocity_(0.0)
{

}

template <typename Scalar, int Dim>
PDMPluginBoundary<Scalar,Dim>::~PDMPluginBoundary()
{

}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar,Dim>::addFixedParticle(unsigned int par_idx)
{
    PDMBase<Scalar,Dim> * driver = dynamic_cast<PDMBase<Scalar,Dim>*>(this->driver_);
    PHYSIKA_ASSERT(driver);
    if(par_idx >= driver->numSimParticles())
    {
        std::cerr<<"particle idx out of range.\n";
        std::exit(EXIT_FAILURE);
    }
    this->fixed_idx_vec_.push_back(par_idx);
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar,Dim>::addEliminateOffsetParticle(unsigned int par_idx)
{
    PDMBase<Scalar,Dim> * driver = dynamic_cast<PDMBase<Scalar,Dim>*>(this->driver_);
    PHYSIKA_ASSERT(driver);
    if(par_idx >= driver->numSimParticles())
    {
        std::cerr<<"particle idx out of range.\n";
        std::exit(EXIT_FAILURE);
    }
    this->eliminate_offset_vec_.push_back(par_idx);

}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar,Dim>::addParticleAdditionalForce(unsigned int par_idx, const Vector<Scalar, Dim> & f)
{
    PDMBase<Scalar,Dim> * driver = dynamic_cast<PDMBase<Scalar,Dim>*>(this->driver_);
    PHYSIKA_ASSERT(driver);
    if(par_idx >= driver->numSimParticles())
    {
        std::cerr<<"particle idx out of range.\n";
        std::exit(EXIT_FAILURE);
    }
    this->force_idx_vec_.push_back(par_idx);
    this->force_vec_.push_back(f);
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar, Dim>::setFloorPos(Scalar pos)
{
    this->floor_pos_ = pos;
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar, Dim>::setRecoveryCoefficient(Scalar recovery_coefficient)
{
    this->recovery_coefficient_ = recovery_coefficient;
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar, Dim>::setEliminateAxis(bool x_axis, bool y_axis, bool z_axis)
{
    this->eliminate_x_offset_ = x_axis;
    this->eliminate_y_offset_ = y_axis;
    this->eliminate_z_offset_ = z_axis;
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar, Dim>::setBulletPos(const Vector<Scalar,Dim> & bullet_pos)
{
    this->bullet_pos_ = bullet_pos;
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar, Dim>::setBulletVelocity(const Vector<Scalar, Dim>& bullet_velocity)
{
    this->bullet_velocity_ = bullet_velocity;
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar, Dim>::setBulletRadius(Scalar bullet_radius)
{
    this->bullet_radius_ = bullet_radius;
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar, Dim>::setBulletKs(Scalar Ks)
{
    this->Ks_ = Ks;
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar, Dim>::setSphereRadius(Scalar sphere_radius)
{
    this->sphere_radius_ = sphere_radius;
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar, Dim>::setSpherePos(const Vector<Scalar,Dim> & sphere_pos)
{
    this->sphere_pos_ = sphere_pos;
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar, Dim>::setSphereVelocity(const Vector<Scalar, Dim> & sphere_velocity)
{
    this->sphere_velocity_ = sphere_velocity;
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar,Dim>::onBeginFrame(unsigned int frame)
{
    // do nothing
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar,Dim>::onEndFrame(unsigned int frame)
{
    // do nothing
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar,Dim>::onBeginTimeStep(Scalar time, Scalar dt)
{
    // add additional force to specified particles
    PDMBase<Scalar,Dim> * driver = dynamic_cast<PDMBase<Scalar,Dim>*>(this->driver_);
    PHYSIKA_ASSERT(driver);
    if (this->enable_force_)
    {
        for (unsigned int i = 0; i < this->force_idx_vec_.size(); i++)
        {
            driver->addParticleForce(this->force_idx_vec_[i], this->force_vec_[i]);
        }
    }

    if (driver->isSimulationPause() == false && this->enable_bullet_)
    {
        bullet_pos_ += bullet_velocity_*dt;
        applyBullet();
    }

    if (driver->isSimulationPause() == false && this->enable_sphere_)
    {
        sphere_pos_ += sphere_velocity_*dt;
        applySphere();
    }
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar,Dim>::enableFix()
{
    this->enable_fix_ = true;
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar,Dim>::disableFix()
{
    this->enable_fix_ = false;
}


template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar,Dim>::enableFloor()
{
    this->enable_floor_ = true;
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar,Dim>::disableFloor()
{
    this->enable_floor_ = false;
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar,Dim>::enableAdditionalForce()
{
    this->enable_force_ = true;
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar,Dim>::disableAdditionalForce()
{
    this->enable_force_ = false;
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar,Dim>::enableEliminateOffset()
{
    this->enable_eliminate_offset_ = true;
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar,Dim>::disableEliminateOffset()
{
    this->enable_eliminate_offset_ = false;
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar,Dim>::enableBullet()
{
    this->enable_bullet_ = true;
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar,Dim>::disableBullet()
{
    this->enable_bullet_ = false;
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar,Dim>::enableSphere()
{
    this->enable_sphere_ = true;
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar,Dim>::disableSphere()
{
    this->enable_sphere_ = false;
}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar,Dim>::onEndTimeStep(Scalar time, Scalar dt)
{
    PDMBase<Scalar,Dim> * driver = dynamic_cast<PDMBase<Scalar,Dim>*>(this->driver_);
    PHYSIKA_ASSERT(driver);
    if (this->enable_floor_)
    {
        unsigned int numParticles = driver->numSimParticles();
        for (unsigned int par_idx = 0; par_idx<numParticles; par_idx++)
        {
            Vector<Scalar, Dim> pos = driver->particleCurrentPosition(par_idx);
            if(pos[1]<this->floor_pos_)
            {
                pos[1] = this->floor_pos_;
                // set particle position and velocity
                driver->setParticleDisplacement(par_idx, pos-driver->particleRestPosition(par_idx));
                Vector<Scalar, Dim> vel = driver->particleVelocity(par_idx);
                vel[1] = -this->recovery_coefficient_*vel[1];
                driver->setParticleVelocity(par_idx,vel);
            }
        }
    }

    if (this->enable_fix_)
    {
        for (unsigned int id =0; id<this->fixed_idx_vec_.size(); id++)
        {
            unsigned int par_idx = this->fixed_idx_vec_[id];
            driver->setParticleDisplacement(par_idx,Vector<Scalar,Dim>(0.0));
            driver->setParticleVelocity(par_idx,Vector<Scalar,Dim>(0.0));
        }
    }

}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar,Dim>::applyBullet()
{
    PDMBase<Scalar,Dim> * driver = dynamic_cast<PDMBase<Scalar,Dim>*>(this->driver_);
    PHYSIKA_ASSERT(driver);

    driver->resetParticleForce();

    Vector<Scalar, Dim> vel_unit = bullet_velocity_;
    vel_unit.normalize();
    unsigned int numParticles = driver->numSimParticles();
    for (unsigned int par_idx = 0; par_idx<numParticles; par_idx++)
    {
        Vector<Scalar,Dim> position = driver->particleCurrentPosition(par_idx);
        Vector<Scalar,Dim> relative_pos =  position-bullet_pos_;
        Scalar len = relative_pos.norm();
        if(len >= 0.75*bullet_radius_ && len <= 1.5*bullet_radius_)
        {
            Vector<Scalar,Dim> project_unit = relative_pos.normalize();
            Vector<Scalar,Dim> force_dir = 0.5*(vel_unit+project_unit);
            force_dir.normalize();

            Vector<Scalar,Dim> force = Ks_*pow((16*(len-0.75*bullet_radius_)*pow(len-1.5*bullet_radius_,2)),2.0/3)*force_dir;
            driver->addParticleForce(par_idx,force);
        }
    }

}

template <typename Scalar, int Dim>
void PDMPluginBoundary<Scalar,Dim>::applySphere()
{
    PDMBase<Scalar,Dim> * driver = dynamic_cast<PDMBase<Scalar,Dim>*>(this->driver_);
    PHYSIKA_ASSERT(driver);

    driver->resetParticleForce();

    Vector<Scalar, Dim> vel_unit = sphere_velocity_;
    unsigned int numParticles = driver->numSimParticles();
    for (unsigned int par_idx = 0; par_idx<numParticles; par_idx++)
    {
        Vector<Scalar,Dim> position = driver->particleCurrentPosition(par_idx);
        Vector<Scalar,Dim> relative_pos =  position-sphere_pos_;
        Scalar len = relative_pos.norm();
        if (len < sphere_radius_)
        {
            relative_pos.normalize();
            //driver->setParticleCurPos(par_idx, sphere_pos_ + sphere_radius_* relative_pos);

            Vector<Scalar, Dim> relative_vel = driver->particleVelocity(par_idx) - sphere_velocity_;
            Scalar relative_vel_norm = relative_vel.norm();
            Scalar cos_theta = relative_vel.dot(relative_pos)/relative_vel_norm;
            if (cos_theta<0)
            {
                Vector<Scalar, Dim> vel_normal = (relative_vel_norm*cos_theta)*relative_pos;
                Vector<Scalar, Dim> vel = sphere_velocity_ - 2*vel_normal + relative_vel;
                driver->setParticleVelocity(par_idx, vel);
            }
        }
    }
}

// explicit instantiations
template class PDMPluginBoundary<double, 3>;
template class PDMPluginBoundary<double, 2>;
template class PDMPluginBoundary<float, 3>;
template class PDMPluginBoundary<float, 2>;


}// end of namespace Physika
