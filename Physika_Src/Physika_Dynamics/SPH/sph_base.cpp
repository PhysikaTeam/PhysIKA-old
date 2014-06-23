/*
 * @file sph_base.cpp 
 * @Basic SPH class,all SPH method inherit from it.
 * @author Sheng Yang
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Dynamics/SPH/sph_base.h"
#include "Physika_Dynamics/Particles/particle.h"


namespace Physika{

template <typename Scalar, int Dim>
SPHBase<Scalar, Dim>::SPHBase():
        sim_itor_(0),
        particle_num_(0),
        reference_density_(0)
{
    dataManager_.addArray("mass", &this->mass_);
    dataManager_.addArray("position", &this->position_);
    dataManager_.addArray("velocity", &this->velocity_);
    dataManager_.addArray("normal", &this->normal_);
}

template <typename Scalar, int Dim>
void SPHBase<Scalar, Dim>::allocMemory(unsigned int particle_num)
{
    this->particle_num_ = particle_num;
    this->mass_.resize(particle_num);
    this->position_.resize(particle_num);
    this->velocity_.resize(particle_num);
    this->normal_.resize(particle_num);
    this->viscous_force_.resize(particle_num);
    this->pressure_force_.resize(particle_num);
    this->surface_force_.resize(particle_num);
    this->volume_.resize(particle_num);
    this->pressure_.resize(particle_num);
    this->density_.resize(particle_num);
}

template <typename Scalar, int Dim>
void SPHBase<Scalar, Dim>::initialize()
{
    initSceneBoundary();
    return ;
}


template <typename Scalar, int Dim>
void SPHBase<Scalar, Dim>::initSceneBoundary()
{

}

template <typename Scalar, int Dim>
void SPHBase<Scalar, Dim>::computeNeighbors()
{

}

template <typename Scalar, int Dim>
void SPHBase<Scalar, Dim>::computeVolume()
{

}

template <typename Scalar, int Dim>
void SPHBase<Scalar, Dim>::computeDensity()
{

}

template <typename Scalar, int Dim>
void SPHBase<Scalar, Dim>::advance(Scalar dt)
{
    //iteration sim_itor begin

    stepEuler(dt);

    //iteration sim_itor end and cost end_time - start_time ;
    this->sim_itor_++;
}

template <typename Scalar, int Dim>
void SPHBase<Scalar, Dim>::stepEuler(Scalar dt)
{

}

template <typename Scalar, int Dim>
void SPHBase<Scalar, Dim>::boundaryHandling()
{
    //TO DO:: handling boundary;
}

template <typename Scalar, int Dim>
SPHBase<Scalar, Dim>::~SPHBase()
{
    this->mass_.release();
    this->position_.release();
    this->velocity_.release();
    this->normal_.release();
    this->viscous_force_.release();
    this->pressure_.release();
    this->surface_force_.release();
    this->volume_.release();
    this->pressure_.release();
    this->density_.release();

}

template <typename Scalar, int Dim>
void SPHBase<Scalar, Dim>::savePositions(std::string in_path, unsigned int in_iter)
{

}

template <typename Scalar, int Dim>
void SPHBase<Scalar, Dim>::saveVelocities(std::string in_path, unsigned int in_iter)
{

}


template class SPHBase<float, 3>;
template class SPHBase<double, 3>;

} //end of namespace Physika
