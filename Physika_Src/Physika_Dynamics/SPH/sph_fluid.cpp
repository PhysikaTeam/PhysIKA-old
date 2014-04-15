/*
 * @file sph_fluid.cpp 
 * @Basic SPH_fluid class, basic fluid simulation uses sph.
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

#include "Physika_Dynamics/SPH/sph_fluid.h"

namespace Physika{

template <typename Scalar, int Dim>
SPHFluid<Scalar, Dim>::SPHFluid()
{
    max_mass_ = 1.0;
    min_mass_ = 1.0;

    initialize();

}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::allocMemory(unsigned int particle_num)
{
    particle_num_ = particle_num;
    SPHBase.allocMemory(particle_num);
    
    phi.resize(particle_num);
    phi.zero();

    energey.resize(particle_num);
    energey.zero();

}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::initialize()
{
    allocMemory(particle_num_);

    // TO DO : need read from config file;
    reference_density_ = 0;
    for (size_t i = 0; i < particle_num_; i++)
    {
        density_[i] = reference_density_;
    }

    time_step_ = 0;
    viscosity_ = 0;
    gravity_ = 0;
    surface_tension_ = 0;
    sampling_distance_ = 0;
    smoothing_length_ = 0;

    //TO DO: set the particle position and velocity;

    computeNeighbors();
    computeDensity();

}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::computeDensity()
{

}


template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::computeNeighbors()
{

}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::computePressure(Scalar dt)
{

}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::computePressureForce(Scalar dt)
{

}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::computeSurfaceTension()
{

}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::computeViscousForce(Scalar dt)
{

}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::computeVolume()
{

}



template <typename Scalar, int Dim>
SPHFluid<Scalar, Dim>::~SPHFluid()
{
    phi.release();
    energey.release();
}

} //end of namespace Physika
