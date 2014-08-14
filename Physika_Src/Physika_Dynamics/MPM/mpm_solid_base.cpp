/*
 * @file mpm_solid_base.cpp 
 * @Brief base class of all MPM drivers for solid.
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

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/MPM/mpm_solid_base.h"
#include "Physika_Dynamics/MPM/MPM_Step_Methods/mpm_solid_step_method_USL.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMSolidBase<Scalar,Dim>::MPMSolidBase()
    :MPMBase<Scalar,Dim>()
{
    this->template setStepMethod<MPMSolidStepMethodUSL<Scalar,Dim> >(); //default step method is USL
}

template <typename Scalar, int Dim>
MPMSolidBase<Scalar,Dim>::MPMSolidBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :MPMBase<Scalar,Dim>(start_frame,end_frame,frame_rate,max_dt,write_to_file)
{
    this->template setStepMethod<MPMSolidStepMethodUSL<Scalar,Dim> >(); //default step method is USL
}

template <typename Scalar, int Dim>
MPMSolidBase<Scalar,Dim>::MPMSolidBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file,
                               const std::vector<SolidParticle<Scalar,Dim>*> &particles)
    :MPMBase<Scalar,Dim>(start_frame,end_frame,frame_rate,max_dt,write_to_file)
{
    setParticles(particles);
}

template <typename Scalar, int Dim>
MPMSolidBase<Scalar,Dim>::~MPMSolidBase()
{
    for(unsigned int i = 0; i < particles_.size(); ++i)
        if(particles_[i])
            delete particles_[i];
}

template <typename Scalar, int Dim>
unsigned int MPMSolidBase<Scalar,Dim>::particleNum() const
{
    return particles_.size();
}

template <typename Scalar, int Dim>
void MPMSolidBase<Scalar,Dim>::addParticle(const SolidParticle<Scalar,Dim> &particle)
{
    SolidParticle<Scalar,Dim> *new_particle = particle.clone();
    particles_.push_back(new_particle);
    unsigned char not_boundary = 0;
    is_bc_particle_.push_back(not_boundary);
}

template <typename Scalar, int Dim>
void MPMSolidBase<Scalar,Dim>::removeParticle(unsigned int particle_idx)
{
    if(particle_idx>=particles_.size())
    {
        std::cerr<<"Warning: MPM particle index out of range, operation ignored!\n";
        return;
    }
    typename std::vector<SolidParticle<Scalar,Dim>*>::iterator iter = particles_.begin() + particle_idx;
    delete *iter; //release memory
    particles_.erase(iter);
    //remove the record in boundary particle
    typename std::vector<unsigned char>::iterator iter2 = is_bc_particle_.begin() + particle_idx;
    is_bc_particle_.erase(iter2);
}

template <typename Scalar, int Dim>
void MPMSolidBase<Scalar,Dim>::setParticles(const std::vector<SolidParticle<Scalar,Dim>*> &particles)
{
    //release data first
    for(unsigned int i = 0; i < particles_.size(); ++i)
        if(particles_[i])
            delete particles_[i];
    particles_.resize(particles.size());
    is_bc_particle_.resize(particles.size());
    //add new particle data
    for(unsigned int i = 0; i < particles.size(); ++i)
    {
        if(particles[i]==NULL)
        {
            std::cerr<<"Warning: pointer to particle "<<i<<" is NULL, ignored!\n";
            continue;
        } 
        particles_[i] = particles[i]->clone();
        is_bc_particle_[i] = 0;
    }
}

template <typename Scalar, int Dim>
const SolidParticle<Scalar,Dim>& MPMSolidBase<Scalar,Dim>::particle(unsigned int particle_idx) const
{
    if(particle_idx>=particles_.size())
    {
        std::cerr<<"Error: MPM particle index out of range, abort program!\n";
        std::exit(EXIT_FAILURE);
    }
    return *particles_[particle_idx];
}

template <typename Scalar, int Dim>
SolidParticle<Scalar,Dim>& MPMSolidBase<Scalar,Dim>::particle(unsigned int particle_idx)
{
    if(particle_idx>=particles_.size())
    {
        std::cerr<<"Error: MPM particle index out of range, abort program!\n";
        std::exit(EXIT_FAILURE);
    }
    return *particles_[particle_idx];
}

template <typename Scalar, int Dim>
const std::vector<SolidParticle<Scalar,Dim>*>& MPMSolidBase<Scalar,Dim>::allParticles() const
{
    return particles_;
}

template <typename Scalar, int Dim>
void MPMSolidBase<Scalar,Dim>::addBCParticle(unsigned int particle_idx)
{
    if(particle_idx>=particles_.size())
    {
        std::cerr<<"Warning: MPM particle index out of range, operation ignored!\n";
        return;
    }
    is_bc_particle_[particle_idx] = 1;
}

template <typename Scalar, int Dim>
void MPMSolidBase<Scalar,Dim>::addBCParticles(const std::vector<unsigned int> &particle_idx)
{
    for(unsigned int i = 0; i < particle_idx.size(); ++i)
        addBCParticle(particle_idx[i]);
}

template <typename Scalar, int Dim>
Scalar MPMSolidBase<Scalar,Dim>::maxParticleVelocityNorm() const
{
    if(particles_.empty())
        return 0;
    Scalar min_vel = (std::numeric_limits<Scalar>::max)();
    for(unsigned int i = 0; i < particles_.size(); ++i)
    {
        Scalar norm_sqr = (particles_[i]->velocity()).normSquared();
        min_vel = norm_sqr < min_vel ? norm_sqr : min_vel;
    }
    return sqrt(min_vel);
}

//explicit instantiations
template class MPMSolidBase<float,2>;
template class MPMSolidBase<float,3>;
template class MPMSolidBase<double,2>;
template class MPMSolidBase<double,3>;

}  //end of namespace Physika
