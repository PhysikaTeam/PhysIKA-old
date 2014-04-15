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
        sim_itor(0),
        particle_num_(0),
        reference_density_(0)
                                
{
    dataManager.addArray("mass", &mass_);
    dataManager.addArray("position", &position_);
    dataManager.addArray("velocity", &velocity_);
    dataManager.addArray("normal", &normal_);

}

template <typename Scalar, int Dim>
void SPHBase<Scalar, Dim>::allocMemory(unsigned int particle_num)
{
    particle_num_ = particle_num;

    mass_.resize(particle_num);
    position_.resize(particle_num);
    velocity_.resize(particle_num);
    normal_.resize(particle_num);
    viscous_force_.resize(particle_num);
    pressure_force_.resize(particle_num);
    surface_force_.resize(particle_num);
    volume_.resize(particle_num);
    pressure_.resize(particle_num);
    density_.resize(particle_num);
    
}

template <typename Scalar, int Dim>
void SPHBase<Scalar, Dim>::initialize()
{
    initSceneBoundary();
    return ;
}

template <typename Scalar, int Dim>
void SPHBase<Scalar, Dim>::computeNeighbors()
{

}

template <typename Scalar, int Dim>
void SPHBase<Scalar, Dim>::advance(Scalar dt)
{
    //iteration sim_itor begin
    clock_t start_time = clock();

    stepEuler(dt);

    clock_t end_time = clock();

    //iteration sim_itor end and cost end_time - start_time ;
    sim_itor++;
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

}

} //end of namespace Physika
