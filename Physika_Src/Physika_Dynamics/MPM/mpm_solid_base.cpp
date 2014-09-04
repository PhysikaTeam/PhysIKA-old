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
    :MPMBase<Scalar,Dim>(),integration_method_(FORWARD_EULER)
{
    this->template setStepMethod<MPMSolidStepMethodUSL<Scalar,Dim> >(); //default step method is USL
}

template <typename Scalar, int Dim>
MPMSolidBase<Scalar,Dim>::MPMSolidBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :MPMBase<Scalar,Dim>(start_frame,end_frame,frame_rate,max_dt,write_to_file),integration_method_(FORWARD_EULER)
{
    this->template setStepMethod<MPMSolidStepMethodUSL<Scalar,Dim> >(); //default step method is USL
}

template <typename Scalar, int Dim>
MPMSolidBase<Scalar,Dim>::MPMSolidBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file,
                               const std::vector<SolidParticle<Scalar,Dim>*> &particles)
    :MPMBase<Scalar,Dim>(start_frame,end_frame,frame_rate,max_dt,write_to_file),integration_method_(FORWARD_EULER)
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
    //append space for the new particle related data
    appendSpaceForParticleRelatedData();
    //set value
    initializeLastParticleRelatedData();
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
    //remove the record in particle related data
    deleteParticleRelatedData(particle_idx);
}

template <typename Scalar, int Dim>
void MPMSolidBase<Scalar,Dim>::setParticles(const std::vector<SolidParticle<Scalar,Dim>*> &particles)
{
    //release data first
    for(unsigned int i = 0; i < particles_.size(); ++i)
        if(particles_[i])
            delete particles_[i];
    particles_.resize(particles.size());
    //set new particle data
    for(unsigned int i = 0; i < particles.size(); ++i)
    {
        if(particles[i]==NULL)
        {
            std::cerr<<"Warning: pointer to particle "<<i<<" is NULL, ignored!\n";
            continue;
        } 
        particles_[i] = particles[i]->clone();
    }
    //resize particle related data according to new particle number 
    allocateSpaceForAllParticleRelatedData();
    //initialize data related to the particles
    initializeAllParticleRelatedData();
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
void MPMSolidBase<Scalar,Dim>::addDirichletParticle(unsigned int particle_idx)
{
    if(particle_idx>=particles_.size())
    {
        std::cerr<<"Warning: MPM particle index out of range, operation ignored!\n";
        return;
    }
    is_dirichlet_particle_[particle_idx] = 1;
}

template <typename Scalar, int Dim>
void MPMSolidBase<Scalar,Dim>::addDirichletParticles(const std::vector<unsigned int> &particle_idx)
{
    for(unsigned int i = 0; i < particle_idx.size(); ++i)
        addDirichletParticle(particle_idx[i]);
}

template <typename Scalar, int Dim>
void MPMSolidBase<Scalar,Dim>::setTimeIntegrationMethod(const IntegrationMethod &method)
{
    integration_method_ = method;
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

template <typename Scalar, int Dim>
void MPMSolidBase<Scalar,Dim>::allocateSpaceForAllParticleRelatedData()
{
    PHYSIKA_ASSERT(this->weight_function_);
    //resize according to new particle number 
    is_dirichlet_particle_.resize(particles_.size());
    particle_initial_volume_.resize(particles_.size());
    //for each particle, preallocate space that can store weight/gradient of maximum
    //number of nodes in range
    unsigned int max_num = 1;
    for(unsigned int i = 0; i < Dim; ++i)
        max_num *= (this->weight_function_->supportRadius())*2+1;
    std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> > max_num_weight_and_gradient_vec(max_num);
    this->particle_grid_weight_and_gradient_.resize(particles_.size(),max_num_weight_and_gradient_vec);
    this->particle_grid_pair_num_.resize(particles_.size(),0);
}

template <typename Scalar, int Dim>
void MPMSolidBase<Scalar,Dim>::initializeAllParticleRelatedData()
{
    for(unsigned int i = 0; i < particleNum(); ++i)
    {
        is_dirichlet_particle_[i] = 0;
        particle_initial_volume_[i] = particles_[i]->volume();
    }
}

template <typename Scalar, int Dim>
void MPMSolidBase<Scalar,Dim>::appendSpaceForParticleRelatedData()
{
    //add space for particle related data
    is_dirichlet_particle_.push_back(0);
    particle_initial_volume_.push_back(0);
    unsigned int max_num = 1;
    for(unsigned int i = 0; i < Dim; ++i)
        max_num *= (this->weight_function_->supportRadius())*2+1;
    std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> > max_num_weight_and_gradient_vec(max_num);
    this->particle_grid_weight_and_gradient_.push_back(max_num_weight_and_gradient_vec);
    this->particle_grid_pair_num_.push_back(0);
}

template <typename Scalar, int Dim>
void MPMSolidBase<Scalar,Dim>::initializeLastParticleRelatedData()
{
    unsigned idx = this->particleNum() - 1;
    is_dirichlet_particle_[idx] = 0;
    particle_initial_volume_[idx] = this->particles_[idx]->volume();
}

template <typename Scalar, int Dim>
void MPMSolidBase<Scalar,Dim>::deleteParticleRelatedData(unsigned int particle_idx)
{
    MPMBase<Scalar,Dim>::deleteParticleRelatedData(particle_idx);
    typename std::vector<Scalar>::iterator iter = particle_initial_volume_.begin() + particle_idx;
    if(iter != particle_initial_volume_.end())
        particle_initial_volume_.erase(iter);
    typename std::vector<unsigned char>::iterator iter2 = is_dirichlet_particle_.begin() + particle_idx;
    if(iter2 != is_dirichlet_particle_.end())
        is_dirichlet_particle_.erase(iter2);
}

//explicit instantiations
template class MPMSolidBase<float,2>;
template class MPMSolidBase<float,3>;
template class MPMSolidBase<double,2>;
template class MPMSolidBase<double,3>;

}  //end of namespace Physika
