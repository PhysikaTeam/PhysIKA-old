/*
 * @file mpm_base.cpp 
 * @Brief Base class of MPM drivers, all MPM methods inherit from it.
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
#include "Physika_Dynamics/MPM/mpm_particle.h"
#include "Physika_Dynamics/MPM/mpm_base.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMBase<Scalar,Dim>::MPMBase()
    :DriverBase<Scalar>()
{
}

template <typename Scalar, int Dim>
MPMBase<Scalar,Dim>::MPMBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :DriverBase<Scalar>(start_frame,end_frame,frame_rate,max_dt,write_to_file)
{
}

template <typename Scalar, int Dim>
MPMBase<Scalar,Dim>::~MPMBase()
{
    for(unsigned int i = 0; i < particles_.size(); ++i)
        if(particles_[i])
            delete particles_[i];
}

template <typename Scalar, int Dim>
unsigned int MPMBase<Scalar,Dim>::particleNum() const
{
    return particles_.size();
}

template <typename Scalar, int Dim>
void MPMBase<Scalar,Dim>::addParticle(const MPMParticle<Scalar,Dim> &particle)
{
    MPMParticle<Scalar,Dim> *new_particle = new MPMParticle<Scalar,Dim>(particle);
    particles_.push_back(new_particle);
}

template <typename Scalar, int Dim>
void MPMBase<Scalar,Dim>::removeParticle(unsigned int particle_idx)
{
    if(particle_idx>=particles_.size())
    {
        std::cerr<<"Error: MPM particle index out of range, abort program!\n";
        std::exit(EXIT_FAILURE);
    }
    typename std::vector<MPMParticle<Scalar,Dim>*>::iterator iter = particles_.begin() + particle_idx;
    particles_.erase(iter);
}

template <typename Scalar, int Dim>
void MPMBase<Scalar,Dim>::setParticles(const std::vector<MPMParticle<Scalar,Dim>*> &particles)
{
    particles_.clear();
    for(unsigned int i = 0; i < particles.size(); ++i)
    {
        if(particles[i]==NULL)
        {
            std::cerr<<"Warning: pointer to particle "<<i<<" is NULL, ignored!\n";
            continue;
        }
        MPMParticle<Scalar,Dim> *mpm_particle = new MPMParticle<Scalar,Dim>(*particles[i]);
        particles_.push_back(mpm_particle);
    }
}

template <typename Scalar, int Dim>
const MPMParticle<Scalar,Dim>& MPMBase<Scalar,Dim>::particle(unsigned int particle_idx) const
{
    if(particle_idx>=particles_.size())
    {
        std::cerr<<"Error: MPM particle index out of range, abort program!\n";
        std::exit(EXIT_FAILURE);
    }
    return *particles_[particle_idx];
}

template <typename Scalar, int Dim>
MPMParticle<Scalar,Dim>& MPMBase<Scalar,Dim>::particle(unsigned int particle_idx)
{
    if(particle_idx>=particles_.size())
    {
        std::cerr<<"Error: MPM particle index out of range, abort program!\n";
        std::exit(EXIT_FAILURE);
    }
    return *particles_[particle_idx];
}

//explicit instantiations
template class MPMBase<float,2>;
template class MPMBase<float,3>;
template class MPMBase<double,2>;
template class MPMBase<double,3>;

}  //end of namespace Physika
