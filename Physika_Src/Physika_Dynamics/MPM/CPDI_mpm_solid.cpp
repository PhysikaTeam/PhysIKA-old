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

#include <limits>
#include <cmath>
#include <iostream>
#include <vector>
#include <map>
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/MPM/mpm_internal.h"
#include "Physika_Dynamics/MPM/MPM_Plugins/mpm_solid_plugin_base.h"
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
    initParticleDomain(particle,domain_corner);
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
        initParticleDomain(*particles[i],domain_corner);
        particle_domain_corners_[i] = domain_corner;
        initial_particle_domain_corners_[i] = domain_corner;
    }
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::updateParticleInterpolationWeight()
{
    //plugin operation
    MPMSolidPluginBase<Scalar,Dim> *plugin = NULL;
    for(unsigned int i = 0; i < this->plugins_.size(); ++i)
    {
        plugin = dynamic_cast<MPMSolidPluginBase<Scalar,Dim>*>(this->plugins_[i]);
        if(plugin)
            plugin->onUpdateParticleInterpolationWeight();
    }

    PHYSIKA_ASSERT(this->particle_grid_weight_and_gradient_.size() == this->particles_.size());
    PHYSIKA_ASSERT(cpdi_update_method_);
    PHYSIKA_ASSERT(this->weight_function_);
    std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> > > &particle_grid_weight_and_gradient = this->particle_grid_weight_and_gradient_;
    std::vector<unsigned int> &particle_grid_pair_num = this->particle_grid_pair_num_;
    const GridWeightFunction<Scalar,Dim> &weight_function = *(this->weight_function_);
    cpdi_update_method_->updateParticleInterpolationWeight(weight_function,particle_grid_weight_and_gradient,particle_grid_pair_num);
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::updateParticleConstitutiveModelState(Scalar dt)
{
    MPMSolid<Scalar,Dim>::updateParticleConstitutiveModelState(dt);
    updateParticleDomain(); //update particle domain after update particle deformation gradient
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::currentParticleDomain(unsigned int particle_idx, ArrayND<Vector<Scalar,Dim>,Dim> &particle_domain_corner) const
{
    if(particle_idx >= this->particles_.size())
    {
        std::cout<<"Warning: particle index out of range, return empty vector!\n";
        particle_domain_corner.clear();
    }
    else
    {
        Vector<unsigned int,Dim> corner_num(2); //2 corners in each dimension
        particle_domain_corner.resize(corner_num);
        for(typename ArrayND<Vector<Scalar,Dim>,Dim>::Iterator iter = particle_domain_corner.begin(); iter != particle_domain_corner.end(); ++iter)
        {
            Vector<unsigned int,Dim> ele_idx = iter.elementIndex();
            unsigned int idx_1d = 0;
            for(unsigned int i = 0; i < Dim; ++i)
            {
                for(unsigned int j = i+1; j < Dim; ++j)
                    ele_idx[i] *= corner_num[j];
                idx_1d += ele_idx[i];
            }
            *iter = particle_domain_corners_[particle_idx][idx_1d];
        }
    }
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::setCurrentParticleDomain(unsigned int particle_idx, const ArrayND<Vector<Scalar,Dim>,Dim> &particle_domain_corner)
{
    if(particle_idx >= this->particles_.size())
        std::cout<<"Warning: particle index out of range, operation ignored!\n";
    else
    {
        PHYSIKA_STATIC_ASSERT(Dim==2||Dim==3,"Invalid dimension specified!");
        unsigned int corner_num = Dim==2 ? 4 : 8;
        unsigned int corner_num_per_dim = 2;
        if(particle_domain_corner.totalElementCount()==corner_num)
        {
            for(typename ArrayND<Vector<Scalar,Dim>,Dim>::ConstIterator iter = particle_domain_corner.begin(); iter != particle_domain_corner.end(); ++iter)
            {
                Vector<unsigned int,Dim> ele_idx = iter.elementIndex();
                unsigned int idx_1d = 0;
                for(unsigned int i = 0; i < Dim; ++i)
                {
                    for(unsigned int j = i+1; j < Dim; ++j)
                        ele_idx[i] *= corner_num_per_dim;
                    idx_1d += ele_idx[i];
                }
                particle_domain_corners_[particle_idx][idx_1d] = *iter;
            }
        }
        else
            std::cout<<"Warning: invalid number of domain corners provided, operation ignored!\n";
    }
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::initialParticleDomain(unsigned int particle_idx, ArrayND<Vector<Scalar,Dim>,Dim> &particle_domain_corner) const
{
    if(particle_idx >= this->particles_.size())
    {
        std::cout<<"Warning: particle index out of range, return empty vector!\n";
        particle_domain_corner.clear();
    }
    else
    {
        Vector<unsigned int,Dim> corner_num(2); //2 corners in each dimension
        particle_domain_corner.resize(corner_num);
        for(typename ArrayND<Vector<Scalar,Dim>,Dim>::Iterator iter = particle_domain_corner.begin(); iter != particle_domain_corner.end(); ++iter)
        {
            Vector<unsigned int,Dim> ele_idx = iter.elementIndex();
            unsigned int idx_1d = 0;
            for(unsigned int i = 0; i < Dim; ++i)
            {
                for(unsigned int j = i+1; j < Dim; ++j)
                    ele_idx[i] *= corner_num[j];
                idx_1d += ele_idx[i];
            }
            *iter = initial_particle_domain_corners_[particle_idx][idx_1d];
        }
    }
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::initializeParticleDomain(unsigned int particle_idx, const ArrayND<Vector<Scalar,Dim>,Dim> &particle_domain_corner)
{
    if(particle_idx >= this->particles_.size())
        std::cout<<"Warning: particle index out of range, operation ignored!\n";
    else
    {
        PHYSIKA_STATIC_ASSERT(Dim==2||Dim==3,"Invalid dimension specified!");
        unsigned int corner_num = Dim==2 ? 4 : 8;
        unsigned int corner_num_per_dim = 2;
        if(particle_domain_corner.totalElementCount()==corner_num)
        {
            for(typename ArrayND<Vector<Scalar,Dim>,Dim>::ConstIterator iter = particle_domain_corner.begin(); iter != particle_domain_corner.end(); ++iter)
            {
                Vector<unsigned int,Dim> ele_idx = iter.elementIndex();
                unsigned int idx_1d = 0;
                for(unsigned int i = 0; i < Dim; ++i)
                {
                    for(unsigned int j = i+1; j < Dim; ++j)
                        ele_idx[i] *= corner_num_per_dim;
                    idx_1d += ele_idx[i];
                }
                particle_domain_corners_[particle_idx][idx_1d] = *iter;
                initial_particle_domain_corners_[particle_idx][idx_1d] = *iter;
            }
        }
        else
            std::cout<<"Warning: invalid number of domain corners provided, operation ignored!\n";
    }
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> CPDIMPMSolid<Scalar,Dim>::currentParticleDomainCorner(unsigned int particle_idx, const Vector<unsigned int,Dim> &corner_idx) const
{
    if(particle_idx >= this->particles_.size())
    {
        std::cout<<"Error: particle index out of range, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    unsigned int corner_num = Dim==2 ? 4 : 8;
    unsigned int corner_num_per_dim = 2;
    Vector<unsigned int,Dim> idx = corner_idx;
    unsigned int idx_1d = 0;
    for(unsigned int i = 0; i < Dim; ++i)
    {
        for(unsigned int j = i+1; j < Dim; ++j)
            idx[i] *= corner_num_per_dim;
        idx_1d += idx[i];
    }
    if(idx_1d >= corner_num)
    {
        std::cout<<"Error: corner index out of range, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    return particle_domain_corners_[particle_idx][idx_1d];
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> CPDIMPMSolid<Scalar,Dim>::initialParticleDomainCorner(unsigned int particle_idx, const Vector<unsigned int,Dim> &corner_idx) const
{
    if(particle_idx >= this->particles_.size())
    {
        std::cout<<"Error: particle index out of range, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    unsigned int corner_num = Dim==2 ? 4 : 8;
    unsigned int corner_num_per_dim = 2;
    Vector<unsigned int,Dim> idx = corner_idx;
    unsigned int idx_1d = 0;
    for(unsigned int i = 0; i < Dim; ++i)
    {
        for(unsigned int j = i+1; j < Dim; ++j)
            idx[i] *= corner_num_per_dim;
        idx_1d += idx[i];
    }
    if(idx_1d >= corner_num)
    {
        std::cout<<"Error: corner index out of range, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    return initial_particle_domain_corners_[particle_idx][idx_1d];
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::allocateSpaceForWeightAndGradient()
{
    PHYSIKA_ASSERT(this->weight_function_);
    PHYSIKA_STATIC_ASSERT(Dim==2||Dim==3,"Wrong dimension specified!");
    //for each particle, allocate space that can store weight/gradient of maximum
    //number of nodes in range of the domain corners
    unsigned int max_num = 1;
    for(unsigned int i = 0; i < Dim; ++i)
        max_num *= (this->weight_function_->supportRadius())*2+1;
    unsigned int corner_num = Dim==2 ? 4 : 8;
    max_num *= corner_num;
    std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> > max_num_weight_and_gradient_vec(max_num);
    this->particle_grid_weight_and_gradient_.resize(this->particles_.size(),max_num_weight_and_gradient_vec);
    this->particle_grid_pair_num_.resize(this->particles_.size(),0);
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::appendSpaceForWeightAndGradient()
{
    PHYSIKA_ASSERT(this->weight_function_);
    PHYSIKA_STATIC_ASSERT(Dim==2||Dim==3,"Wrong dimension specified!");
    unsigned int max_num = 1;
    for(unsigned int i = 0; i < Dim; ++i)
        max_num *= (this->weight_function_->supportRadius())*2+1;
    unsigned int corner_num = Dim==2 ? 4 : 8;
    max_num *= corner_num;
    std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> > max_num_weight_and_gradient_vec(max_num);
    this->particle_grid_weight_and_gradient_.push_back(max_num_weight_and_gradient_vec);
    this->particle_grid_pair_num_.push_back(0);
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::initParticleDomain(const SolidParticle<Scalar,2> &particle,
                                                std::vector<Vector<Scalar,2> > &domain_corner)
{
    //determine the position of the corners via particle volume and position
    unsigned int corner_num = 4;
    domain_corner.resize(corner_num);
    Scalar particle_radius = sqrt(particle.volume())/2.0;//assume the particle occupies rectangle space
    PHYSIKA_ASSERT(Dim==2);
    Vector<Scalar,2> min_corner = particle.position() - Vector<Scalar,2>(particle_radius);
    Vector<Scalar,2> bias(0);
    domain_corner[0] = min_corner;
    bias[1] = 2*particle_radius;
    domain_corner[1] = min_corner + bias;
    bias[0] = 2*particle_radius;
    bias[1] = 0;
    domain_corner[2] = min_corner + bias;
    bias[1] = 2*particle_radius;
    domain_corner[3] = min_corner + bias;
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::initParticleDomain(const SolidParticle<Scalar,3> &particle,
                                                std::vector<Vector<Scalar,3> > &domain_corner)
{
    //determine the position of the corners via particle volume and position
    unsigned int corner_num = 8;
    domain_corner.resize(corner_num);
    Scalar particle_radius = pow(particle.volume(),1.0/3.0)/2.0;//assume the particle occupies cubic space
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
    PHYSIKA_ASSERT(cpdi_update_method_);
    cpdi_update_method_->updateParticleDomain();
}

//explicit instantiations
template class CPDIMPMSolid<float,2>;
template class CPDIMPMSolid<float,3>;
template class CPDIMPMSolid<double,2>;
template class CPDIMPMSolid<double,3>;

}  //end of namespace Physika
