/*
 * @file mpm_isotropic_hyperelastic_solid.cpp
 * @Brief MPM driver used to simulate hyperelastic solid.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Dynamics/Driver/driver_plugin_base.h"
#include "Physika_Dynamics/Particles/isotropic_hyperelastic_particle.h"
#include "Physika_Dynamics/MPM/mpm_isotropic_hyperelastic_solid.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMIsotropicHyperelasticSolid<Scalar,Dim>::MPMIsotropicHyperelasticSolid()
    :DriverBase<Scalar>()
{
}

template <typename Scalar, int Dim>
MPMIsotropicHyperelasticSolid<Scalar,Dim>::MPMIsotropicHyperelasticSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :DriverBase<Scalar>(start_frame,end_frame,frame_rate,max_dt,write_to_file)
{
}

template <typename Scalar, int Dim>
MPMIsotropicHyperelasticSolid<Scalar,Dim>::~MPMIsotropicHyperelasticSolid()
{
    for(unsigned int i = 0; i < particles_.size(); ++i)
        if(particles_[i])
            delete particles_[i];
}

template <typename Scalar, int Dim>
unsigned int MPMIsotropicHyperelasticSolid<Scalar,Dim>::particleNum() const
{
    return particles_.size();
}

template <typename Scalar, int Dim>
void MPMIsotropicHyperelasticSolid<Scalar,Dim>::addParticle(const IsotropicHyperelasticParticle<Scalar,Dim> &particle)
{
    IsotropicHyperelasticParticle<Scalar,Dim> *new_particle = particle.clone();
    particles_.push_back(new_particle);
}

template <typename Scalar, int Dim>
void MPMIsotropicHyperelasticSolid<Scalar,Dim>::removeParticle(unsigned int particle_idx)
{
    if(particle_idx>=particles_.size())
    {
        std::cerr<<"Error: MPM particle index out of range, abort program!\n";
        std::exit(EXIT_FAILURE);
    }
    typename std::vector<IsotropicHyperelasticParticle<Scalar,Dim>*>::iterator iter = particles_.begin() + particle_idx;
    delete *iter; //release memory
    particles_.erase(iter);
}

template <typename Scalar, int Dim>
void MPMIsotropicHyperelasticSolid<Scalar,Dim>::setParticles(const std::vector<IsotropicHyperelasticParticle<Scalar,Dim>*> &particles)
{
    particles_.clear();
    for(unsigned int i = 0; i < particles.size(); ++i)
    {
        if(particles[i]==NULL)
        {
            std::cerr<<"Warning: pointer to particle "<<i<<" is NULL, ignored!\n";
            continue;
        }
        IsotropicHyperelasticParticle<Scalar,Dim> *mpm_particle = particles[i]->clone();
        particles_.push_back(mpm_particle);
    }
}

template <typename Scalar, int Dim>
const IsotropicHyperelasticParticle<Scalar,Dim>& MPMIsotropicHyperelasticSolid<Scalar,Dim>::particle(unsigned int particle_idx) const
{
    if(particle_idx>=particles_.size())
    {
        std::cerr<<"Error: MPM particle index out of range, abort program!\n";
        std::exit(EXIT_FAILURE);
    }
    return *particles_[particle_idx];
}

template <typename Scalar, int Dim>
IsotropicHyperelasticParticle<Scalar,Dim>& MPMIsotropicHyperelasticSolid<Scalar,Dim>::particle(unsigned int particle_idx)
{
    if(particle_idx>=particles_.size())
    {
        std::cerr<<"Error: MPM particle index out of range, abort program!\n";
        std::exit(EXIT_FAILURE);
    }
    return *particles_[particle_idx];
}

//explicit instantiations
template class MPMIsotropicHyperelasticSolid<float,2>;
template class MPMIsotropicHyperelasticSolid<float,3>;
template class MPMIsotropicHyperelasticSolid<double,2>;
template class MPMIsotropicHyperelasticSolid<double,3>;

}  //end of namespace Physika
