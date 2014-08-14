/*
 * @file CPDI_mpm_solid.cpp 
 * @Brief CPDI(CPDI2) MPM driver used to simulate solid, uniform grid.
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

#include <cmath>
#include <iostream>
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/MPM/CPDI_mpm_solid.h"

namespace Physika{

template <typename Scalar, int Dim>
CPDIMPMSolid<Scalar,Dim>::CPDIMPMSolid()
    :MPMSolid<Scalar,Dim>(),cpdi_update_method_(NULL)
{
    setCPDIUpdateMethod<CPDIUpdateMethod<Scalar,Dim> >();
}

template <typename Scalar, int Dim>
CPDIMPMSolid<Scalar,Dim>::CPDIMPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :MPMSolid<Scalar,Dim>(start_frame,end_frame,frame_rate,max_dt,write_to_file),cpdi_update_method_(NULL)
{
    setCPDIUpdateMethod<CPDIUpdateMethod<Scalar,Dim> >();
}

template <typename Scalar, int Dim>
CPDIMPMSolid<Scalar,Dim>::CPDIMPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file,
                                       const std::vector<SolidParticle<Scalar,Dim>*> &particles, const Grid<Scalar,Dim> &grid)
    :MPMSolid<Scalar,Dim>(start_frame,end_frame,frame_rate,max_dt,write_to_file,particles,grid),cpdi_update_method_(NULL)
{
    setCPDIUpdateMethod<CPDIUpdateMethod<Scalar,Dim> >();
}

template <typename Scalar, int Dim>
CPDIMPMSolid<Scalar,Dim>::~CPDIMPMSolid()
{
    if(cpdi_update_method_)
        delete cpdi_update_method_;
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::addParticle(const SolidParticle<Scalar,Dim> &particle)
{
    MPMSolid<Scalar,Dim>::addParticle(particle);
    //add space for particle domain corners
    std::vector<Vector<Scalar,Dim> > domain_corner;
    //determine the position of the corners via particle volume and position
    DimensionTrait<Dim> trait;
    initParticleDomain(particle,domain_corner,trait);
    particle_domain_corners_.push_back(domain_corner); 
    initial_particle_domain_corners_.push_back(domain_corner);
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::removeParticle(unsigned int particle_idx)
{
    MPMSolid<Scalar,Dim>::removeParticle(particle_idx);
    typename std::vector<std::vector<Vector<Scalar,Dim> > >::iterator iter = particle_domain_corners_.begin() + particle_idx;
    particle_domain_corners_.erase(iter);
    typename std::vector<std::vector<Vector<Scalar,Dim> > >::iterator iter2 = initial_particle_domain_corners_.begin() + particle_idx;
    initial_particle_domain_corners_.erase(iter2);
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::setParticles(const std::vector<SolidParticle<Scalar,Dim>*> &particles)
{
    MPMSolid<Scalar,Dim>::setParticles(particles);
    particle_domain_corners_.resize(particles.size());
    initial_particle_domain_corners_.resize(particles.size());
    for(unsigned int i = 0; i < particle_domain_corners_.size(); ++i)
    {
        std::vector<Vector<Scalar,Dim> > domain_corner;
        //determine the position of the corners via particle volume and position
        DimensionTrait<Dim> trait;
        initParticleDomain(*particles[i],domain_corner,trait);
        particle_domain_corners_[i] = domain_corner;
        initial_particle_domain_corners_[i] = domain_corner;
    }
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::updateParticleInterpolationWeight()
{
//TO DO
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::updateParticleConstitutiveModelState(Scalar dt)
{
    MPMSolid<Scalar,Dim>::updateParticleConstitutiveModelState(dt);
    updateParticleDomain(); //update particle domain after update particle deformation gradient
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::currentParticleDomain(unsigned int particle_idx, std::vector<Vector<Scalar,Dim> > &particle_domain_corner)
{
    if(particle_idx >= this->particles_.size())
    {
        std::cout<<"Warning: particle index out of range, return empty vector!\n";
        particle_domain_corner.clear();
    }
    else
        particle_domain_corner = particle_domain_corners_[particle_idx];
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::setCurrentParticleDomain(unsigned int particle_idx, const std::vector<Vector<Scalar,Dim> > &particle_domain_corner)
{
    if(particle_idx >= this->particles_.size())
        std::cout<<"Warning: particle index out of range, operation ignored!\n";
    else
    {
        PHYSIKA_ASSERT(Dim==2||Dim==3);
        unsigned int corner_num = Dim==2 ? 4 : 8;
        if(particle_domain_corner.size()==corner_num)
            particle_domain_corners_[particle_idx] = particle_domain_corner;
        else
            std::cout<<"Warning: invalid number of domain corners provided, operation ignored!\n";
    }
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::initialParticleDomain(unsigned int particle_idx, std::vector<Vector<Scalar,Dim> > &particle_domain_corner)
{
    if(particle_idx >= this->particles_.size())
    {
        std::cout<<"Warning: particle index out of range, return empty vector!\n";
        particle_domain_corner.clear();
    }
    else
        particle_domain_corner = initial_particle_domain_corners_[particle_idx];
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::initializeParticleDomain(unsigned int particle_idx, const std::vector<Vector<Scalar,Dim> > &particle_domain_corner)
{
    if(particle_idx >= this->particles_.size())
        std::cout<<"Warning: particle index out of range, operation ignored!\n";
    else
    {
        PHYSIKA_ASSERT(Dim==2||Dim==3);
        unsigned int corner_num = Dim==2 ? 4 : 8;
        if(particle_domain_corner.size()==corner_num)
        {
            particle_domain_corners_[particle_idx] = particle_domain_corner;
            initial_particle_domain_corners_[particle_idx] = particle_domain_corner;
        }
        else
            std::cout<<"Warning: invalid number of domain corners provided, operation ignored!\n";
    }
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::initParticleDomain(const SolidParticle<Scalar,2> &particle,
                                                std::vector<Vector<Scalar,2> > &domain_corner, DimensionTrait<2> trait)
{
    //determine the position of the corners via particle volume and position
    unsigned int corner_num = 4;
    domain_corner.resize(corner_num);
    Scalar particle_radius = sqrt(particle.volume()/(4.0*PI));
    PHYSIKA_ASSERT(Dim==2);
    Vector<Scalar,2> min_corner = particle.position() - Vector<Scalar,2>(particle_radius);
    Vector<Scalar,2> bias(0);
    domain_corner[0] = min_corner;
    bias[0] = 2*particle_radius;
    domain_corner[1] = min_corner + bias;
    bias[1] = 2*particle_radius;
    domain_corner[2] = min_corner + bias;
    bias[0] = 0;
    domain_corner[3] = min_corner + bias;
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::initParticleDomain(const SolidParticle<Scalar,3> &particle,
                                                std::vector<Vector<Scalar,3> > &domain_corner, DimensionTrait<3> trait)
{
    //determine the position of the corners via particle volume and position
    unsigned int corner_num = 8;
    domain_corner.resize(corner_num);
    Scalar particle_radius = pow(particle.volume()*3/(4*PI),1.0/3.0);
    PHYSIKA_ASSERT(Dim==3);
    Vector<Scalar,3> min_corner = particle.position() - Vector<Scalar,3>(particle_radius);
    Vector<Scalar,3> bias(0);
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            for(unsigned int k = 0; k < 2; ++k)
            {
                bias[0] = i*2*particle_radius;
                bias[1] = j*2*particle_radius;
                bias[2] = k*2*particle_radius;
                domain_corner[i*2*2+j*2+k] = min_corner + bias;
            }
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::updateParticleDomain()
{
//TO DO
}

//explicit instantiations
template class CPDIMPMSolid<float,2>;
template class CPDIMPMSolid<float,3>;
template class CPDIMPMSolid<double,2>;
template class CPDIMPMSolid<double,3>;

}  //end of namespace Physika
