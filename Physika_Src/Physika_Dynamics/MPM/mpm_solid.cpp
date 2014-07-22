/*
 * @file mpm_solid.cpp
 * @Brief MPM driver used to simulate solid.
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
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/MPM/mpm_solid.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMSolid<Scalar,Dim>::MPMSolid()
    :DriverBase<Scalar>()
{
}

template <typename Scalar, int Dim>
MPMSolid<Scalar,Dim>::MPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :DriverBase<Scalar>(start_frame,end_frame,frame_rate,max_dt,write_to_file)
{
}

template <typename Scalar, int Dim>
MPMSolid<Scalar,Dim>::MPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file,
                               const std::vector<SolidParticle<Scalar,Dim>*> &particles, const Grid<Scalar,Dim> &grid)
    :DriverBase<Scalar>(start_frame,end_frame,frame_rate,max_dt,write_to_file),grid_(grid)
{
    setParticles(particles);
    synchronizeGridData();
}

template <typename Scalar, int Dim>
MPMSolid<Scalar,Dim>::~MPMSolid()
{
    for(unsigned int i = 0; i < particles_.size(); ++i)
        if(particles_[i])
            delete particles_[i];
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::initConfiguration(const std::string &file_name)
{
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::advanceStep(Scalar dt)
{
//TO DO
}

template <typename Scalar, int Dim>
Scalar MPMSolid<Scalar,Dim>::computeTimeStep()
{
//TO DO
    return 0;
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::addPlugin(DriverPluginBase<Scalar> *plugin)
{
//TO DO
}

template <typename Scalar, int Dim>
bool MPMSolid<Scalar,Dim>::withRestartSupport() const
{
    return false;
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::write(const std::string &file_name)
{
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::read(const std::string &file_name)
{
//TO DO
}

template <typename Scalar, int Dim>
unsigned int MPMSolid<Scalar,Dim>::particleNum() const
{
    return particles_.size();
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::addParticle(const SolidParticle<Scalar,Dim> &particle)
{
    SolidParticle<Scalar,Dim> *new_particle = particle.clone();
    particles_.push_back(new_particle);
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::removeParticle(unsigned int particle_idx)
{
    if(particle_idx>=particles_.size())
    {
        std::cerr<<"Error: MPM particle index out of range, abort program!\n";
        std::exit(EXIT_FAILURE);
    }
    typename std::vector<SolidParticle<Scalar,Dim>*>::iterator iter = particles_.begin() + particle_idx;
    delete *iter; //release memory
    particles_.erase(iter);
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::setParticles(const std::vector<SolidParticle<Scalar,Dim>*> &particles)
{
    //release data first
    for(unsigned int i = 0; i < particles_.size(); ++i)
        if(particles_[i])
            delete particles_[i];
    particles_.clear();
    //add new particle data
    for(unsigned int i = 0; i < particles.size(); ++i)
    {
        if(particles[i]==NULL)
        {
            std::cerr<<"Warning: pointer to particle "<<i<<" is NULL, ignored!\n";
            continue;
        }
        SolidParticle<Scalar,Dim> *mpm_particle = particles[i]->clone();
        particles_.push_back(mpm_particle);
    }
}

template <typename Scalar, int Dim>
const SolidParticle<Scalar,Dim>& MPMSolid<Scalar,Dim>::particle(unsigned int particle_idx) const
{
    if(particle_idx>=particles_.size())
    {
        std::cerr<<"Error: MPM particle index out of range, abort program!\n";
        std::exit(EXIT_FAILURE);
    }
    return *particles_[particle_idx];
}

template <typename Scalar, int Dim>
SolidParticle<Scalar,Dim>& MPMSolid<Scalar,Dim>::particle(unsigned int particle_idx)
{
    if(particle_idx>=particles_.size())
    {
        std::cerr<<"Error: MPM particle index out of range, abort program!\n";
        std::exit(EXIT_FAILURE);
    }
    return *particles_[particle_idx];
}

template <typename Scalar, int Dim>
const Grid<Scalar,Dim>& MPMSolid<Scalar,Dim>::grid() const
{
    return grid_;
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::setGrid(const Grid<Scalar,Dim> &grid)
{
    grid_ = grid;
    synchronizeGridData();
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::initialize()
{
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::synchronizeGridData()
{
    Vector<unsigned int,Dim> node_num = grid_.nodeNum();
    for(unsigned int i = 0; i < Dim; ++i)
    {
        grid_mass_.resize(node_num[i],i);
        grid_velocity_.resize(node_num[i],i);
    }
}

//explicit instantiations
template class MPMSolid<float,2>;
template class MPMSolid<float,3>;
template class MPMSolid<double,2>;
template class MPMSolid<double,3>;

}  //end of namespace Physika
