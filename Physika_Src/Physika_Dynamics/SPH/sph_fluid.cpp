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
#include "Physika_Dynamics/SPH/sph_kernel.h"


namespace Physika{

template <typename Scalar, int Dim>
SPHFluid<Scalar, Dim>::SPHFluid()
{
    this->max_mass_ = 1.0;
    this->min_mass_ = 1.0;

    initialize();

}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::allocMemory(unsigned int particle_num)
{
    this->particle_num_ = particle_num;
    SPHBase<Scalar,Dim>::allocMemory(particle_num);
    
    this->phi_.resize(particle_num);
    this->phi_.zero();

    this->energy_.resize(particle_num);
    this->energy_.zero();

    this->neighbor_lists_.resize(particle_num);
    this->neighbor_lists_.zero();

    this->small_density_.resize(particle_num);
    this->small_density_.zero();

    this->small_scale_.resize(particle_num);
    this->small_scale_.zero();
}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::initConfiguration()
{
    this->config_file_.addOptionOptional("timestep", &(this->time_step_), static_cast<Scalar>(0.001));
    this->config_file_.addOptionOptional("viscosity", &(this->viscosity_), static_cast<Scalar>(280000));
    this->config_file_.addOptionOptional("surfacetension",&(this->surface_tension_),static_cast<Scalar>(54000));
    this->config_file_.addOptionOptional("density", &(this->reference_density_), static_cast<Scalar>(1000));
    this->config_file_.addOptionOptional("gravity", &(this->gravity_), static_cast<Scalar>(-9.8));
    this->config_file_.addOptionOptional("sampling_distance", &(this->sampling_distance_), static_cast<Scalar>(0.005));
    this->config_file_.addOptionOptional("smoothing_length", &(this->smoothing_length_), static_cast<Scalar>(2.5));
    this->config_file_.addOptionOptional("init_from_file", &(this->init_from_file_), false);
    this->config_file_.addOptionOptional("init_file_name", &(this->init_file_name_), static_cast<std::string>(""));
    this->config_file_.addOptionOptional("x_num", &(this->x_num_), (100));
    this->config_file_.addOptionOptional("y_num", &(this->y_num_), (100));
    this->config_file_.addOptionOptional("z_num", &(this->z_num_), (1));

    if(!this->init_from_file_)
        this->particle_num_ = x_num_ * y_num_ *z_num_;
    else
    {
        this->particle_num_ = 0;
    }

}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::initScene()
{
    for (unsigned int i = 0; i < this->particle_num_; i++)
    {
        this->density_[i] = this->reference_density_;
    }

    for ( int i = 0; i < x_num_; i++)
    {
        for ( int j = 0; j < y_num_; j++)
        {
            for( int k = 0; k < z_num_; k++)
            {
                if(Dim == 3)
                {
                    unsigned id = i + j*x_num_ + k*x_num_*y_num_;
                    Vector<Scalar, Dim> position(i*(this->sampling_distance_), j*(this->sampling_distance_), k*(this->sampling_distance_));
                    (this->position_)[id] = position;
                    (this->velocity_)[id] = Vector<Scalar, Dim>(0);
                    (this->mass_)[id] = 1;
                }
            }
        } 
    }
}


template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::initialize()
{
    initConfiguration();

    initSceneBoundary();

    allocMemory(this->particle_num_);

    initScene();

    //computeNeighbors();

    //computeDensity();

    //computeMass();

    //computeNeighbors();

    //computeDensity();
}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::computeMass()
{
    Scalar max_large_density = 0;
    for (unsigned int i = 0; i < this->particle_num_; i++)
    {
        if((this->density_)[i] > max_large_density) max_large_density = (this->density_)[i];
    }
    Scalar ratio_large = (this->reference_density_) / max_large_density;
    max_mass_ *= ratio_large;

    for (unsigned int i = 0; i < this->particle_num_; i++)
    {
        (this->mass_)[i] = max_mass_;
    }
}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::computeDensity()
{
    SPH_Kernel<Scalar> &kernel = KernelFactory<Scalar>::createKernel(KernelFactory<Scalar>::Spiky);
    for (unsigned int i = 0; i < this->particle_num_; i++)
    {
        NeighborList<Scalar> & neighborlist_i = this->neighbor_lists_[i];
        int size_i = neighborlist_i.size_;
        Scalar tmp = 0.0;
        for (int j = 0; j < size_i; j++)
        {
            Scalar r = neighborlist_i.distance_[j];
            tmp += kernel.weight(r, max_length_);
        }
        this->density_[i] = tmp;
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
    
    this->pressure_.zero();
    for (unsigned int i = 0; i < this->particle_num_; i++)
    {
        //TO DO: mod the compute formula
        this->pressure_[i] = (this->density_[i] - this->reference_density_);

        if(this->pressure_[i] < 0.0) 
            this->pressure_[i] = 0.0f;
    }
    
}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::computePressureForce(Scalar dt)
{
     SPH_Kernel<Scalar> &kernel = KernelFactory<Scalar>::createKernel(KernelFactory<Scalar>::Spiky);
    for (unsigned int i = 0; i < this->particle_num_; i++)
    {
        NeighborList<Scalar> & neighborlist_i = this->neighbor_lists_[i];
        int size_i = neighborlist_i.size_;
        Scalar v_i = this->volume_[i];
        for (int ne = 0; ne < size_i; ne++)
        {
            Scalar d_kernel = 0.0;
            Scalar r = neighborlist_i.distance_[ne];
            int j = neighborlist_i.ids_[ne];
            
            Scalar v_j = this->volume_[j];
            Vector<Scalar, Dim> f_t =  0.5*v_i*v_j*kernel.gradient(r, max_length_)*(this->position_[j] - this->position_[i]) * (1.0f/r);
            this->pressure_force_[i] += f_t;
            this->pressure_force_[j] -= f_t;
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
    SPH_Kernel<Scalar>& kernel = KernelFactory<Scalar>::createKernel(KernelFactory<Scalar>::Laplacian);
    this->viscous_force_.zero();
    for (unsigned int i = 0; i < this->particle_num_; i++)
    {
        NeighborList<Scalar>& neighborlist_i = this->neighbor_lists_[i];
        int size_i = neighborlist_i.size_;
        Scalar v_i = this->volume_[i];
        for ( int ne = 0; ne < size_i; ne++ )
        {
            int j = neighborlist_i.ids_[ne];

            Scalar r = neighborlist_i.distance_[ne];
            Scalar v_j = this->volume_[j];
            Vector<Scalar, Dim> f_t = 0.5f*v_i*v_j*kernel.weight(r, max_length_)*(this->velocity_[j]-this->velocity_[i]);
            this->viscous_force_[i] += f_t;
            this->viscous_force_[j] -= f_t;
        }
 
    }

}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::computeVolume()
{
    for (unsigned int i = 0; i < this->particle_num_; i++)
    {
        this->volume_[i] = this->mass_[i] / this->density_[i];
    }
    
}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::advect(Scalar dt)
{
    for (unsigned int i = 0; i < this->particle_num_; i++)
    {
        this->velocity_[i] += dt * (this->mass_[i])*(this->viscous_force_[i] + this->pressure_force_[i] + Vector<Scalar, Dim>(0,this->gravity_,0));
        this->position_[i] += dt * this->velocity_[i];
    }
}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::stepEuler(Scalar dt)
{
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
}


template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::initSceneBoundary()
{

}

template <typename Scalar, int Dim>
void SPHFluid<Scalar, Dim>::advance(Scalar dt)
{
    //iteration sim_itor begin

    stepEuler(dt);

    //iteration sim_itor end and cost end_time - start_time ;
    this->sim_itor_++;
}

template <typename Scalar, int Dim>
Scalar SPHFluid<Scalar, Dim>::getTimeStep()
{
    return this->time_step_;
}

template class SPHFluid<float, 3>;
template class SPHFluid<double ,3>;
} //end of namespace Physika


