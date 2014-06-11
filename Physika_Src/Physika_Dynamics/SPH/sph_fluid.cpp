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
#include "Physika_Dynamics/sph/sph_kernel.h"


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
    this->particle_num_ = particle_num;
    SPHBase<Scalar,Dim>::allocMemory(particle_num);
    
    this->phi_.resize(particle_num);
    this->phi_.zero();

    this->energey_.resize(particle_num);
    this->energey_.zero();

   // this->neighborLists_.resize(particle_num);
    //this->neighborLists_.zero();
}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::initialize()
{
    allocMemory(this->particle_num_);

    // TO DO : need read from config file;
    this->reference_density_ = 0;
    for (size_t i = 0; i < this->particle_num_; i++)
    {
        this->density_[i] = this->reference_density_;
    }

    this->time_step_ = 0;
    this->viscosity_ = 0;
    this->gravity_ = 0;
    this->surface_tension_ = 0;
    this->sampling_distance_ = 0;
    this->smoothing_length_ = 0;

    //TO DO: set the particle position and velocity;

    computeNeighbors();
    computeDensity();

}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::computeDensity()
{
    SPH_Kernel<Scalar> &kernel = KernelFactory::CreateKernel(KernelFactory::Spiky);
    for (int i = 0; i < N; i++)
    {
        NeighborList & neighborlist_i = neighborLists[i];
        int size_i = neighborlist_i.size;
        Scalar tmp = 0.0;
        for (int j = 0; j < size_i; j++)
        {
            Scalar r = neighborlist_i.distance[j];
            r += kernel.weight(r, max_length_);
        }
        density_[i] = r;
    }
       
}


template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::computeNeighbors()
{
    //TO DO: compute new neighbors froe new positions;
}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::computePressure(Scalar dt)
{
    
    pressure_.zero();
    for (int i = 0; i < N; i++)
    {
        //TO DO: mod the compute formula
        pressure_[i] = (density_[i] - reference_density_);

        if(pressure_[i] < 0.0) 
            pressure_[i] = 0.0f;
    }
    
}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::computePressureForce(Scalar dt)
{
     SPH_Kernel<Scalar> &kernel = KernelFactory::CreateKernel(KernelFactory::Spiky);
    for (int i = 0; i < N; i++)
    {
        NeighborList & neighborlist_i = neighborLists[i];
        int size_i = neighborlist_i.size;
        Scalar v_i = volume_[i];
        for (int ne = 0; ne < size_i; ne++)
        {
            Scalar d_kernel = 0.0;
            Scalar r = neighborlist_i.distance[ne];
            int j = neighborlist_i.ids[ne];
            
            Scalar v_j = volume_[j];
            Vector<Scalar, Dim> f_t =  0.5*v_i*v_j*kernel.gradient(r, max_length_)*(position_[j]-position_[i]) * (1.0f/r);
            pressure_force_[i] += f_t;
            pressure_force_[j] -= f_t;
        }
    }
}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::computeSurfaceTension()
{
    //TO DO: compute surface Tension in high level sph, here can be null; 
}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::computeViscousForce(Scalar dt)
{
    SPH_Kernel& kernel = KernelFactory::CreateKernel(KernelFactory::Laplacian);
    viscous_force_.zero();
    for (int i = 0; i < N; i++)
    {
        NeighborList& neighborlist_i = neighborLists[i];
        int size_i = neighborlist_i.size;
        Scalar v_i = volArr[i];
        for ( int ne = 0; ne < size_i; ne++ )
        {
            int j = neighborlist_i.ids[ne];

            Scalar r = neighborlist_i.distance[ne];
            Scalar v_j = volArr[j];
            Vector<Scalar, Dim> f_t = 0.5f*v_i*v_j*kernel.Weight(r, max_length_)*(velocity_[j]-velocity_[i]);
            viscous_force_[i] += f_t;
            viscous_force_[j] -= f_t;
        }
 
    }

}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::computeVolume()
{
    for (int i = 0; i < N; i++)
    {
        volume_[i] = mass_[i] / density_[i];
    }
    
}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::advect(Scalar dt)
{
    for (int i = 0; i < N; i++)
    {
        velocity_[i] += dt/mass_[i]*(viscous_force_[i] + pressure_force_[i] + gravity_);
        position_[i] += dt * velocity_[i];
    }
}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::stepEuler(Scalar dt)
{
    boundaryHandling();

    computeNeighbors();

    computeDensity();

    computeVolume();

    computeViscousForce(dt);

    computePressure(dt);

    computePressureForce(dt);

    advect(dt);
}

template <typename Scalar, int Dim>
SPHFluid<Scalar, Dim>::~SPHFluid()
{
    this->phi_.release();
    this->energy_.release();
}

} //end of namespace Physika
