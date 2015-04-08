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
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/MPM/mpm_internal.h"
#include "Physika_Dynamics/MPM/MPM_Plugins/mpm_solid_plugin_base.h"
#include "Physika_Dynamics/MPM/CPDI_Update_Methods/CPDI2_update_method.h"
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
                                       const Grid<Scalar,Dim> &grid)
    :MPMSolid<Scalar,Dim>(start_frame,end_frame,frame_rate,max_dt,write_to_file,grid),cpdi_update_method_(NULL)
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
bool CPDIMPMSolid<Scalar,Dim>::withRestartSupport() const
{
    return false;
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::write(const std::string &file_name)
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::read(const std::string &file_name)
{
    throw PhysikaException("Not implemented!");
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
    const GridWeightFunction<Scalar,Dim> &weight_function = *(this->weight_function_);
    CPDI2UpdateMethod<Scalar,Dim> *update_method = dynamic_cast<CPDI2UpdateMethod<Scalar,Dim>*>(cpdi_update_method_);
    if(update_method)  //CPDI2
        update_method->updateParticleInterpolationWeight(weight_function,this->particle_grid_weight_and_gradient_,this->particle_grid_pair_num_,
                                                         corner_grid_weight_,corner_grid_pair_num_);
    else //CPDI
        cpdi_update_method_->updateParticleInterpolationWeight(weight_function,this->particle_grid_weight_and_gradient_,this->particle_grid_pair_num_);
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::updateParticlePosition(Scalar dt)
{
    CPDI2UpdateMethod<Scalar,Dim> *update_method = dynamic_cast<CPDI2UpdateMethod<Scalar,Dim>*>(cpdi_update_method_);
    if(update_method)  //CPDI2
    {
        //plugin operation
        MPMSolidPluginBase<Scalar,Dim> *plugin = NULL;
        for(unsigned int i = 0; i < this->plugins_.size(); ++i)
        {
            plugin = dynamic_cast<MPMSolidPluginBase<Scalar,Dim>*>(this->plugins_[i]);
            if(plugin)
                plugin->onUpdateParticlePosition(dt);
        }
        //update particle domain before update particle position
        update_method->updateParticleDomain(corner_grid_weight_,corner_grid_pair_num_,dt);
        //update particle position with CPDI2
        update_method->updateParticlePosition(dt,this->is_dirichlet_particle_);
    }
    else //CPDI
    {
        //update particle domain before update particle position
        cpdi_update_method_->updateParticleDomain();
        MPMSolid<Scalar,Dim>::updateParticlePosition(dt);
    }
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::currentParticleDomain(unsigned int object_idx, unsigned int particle_idx,
                                                             ArrayND<Vector<Scalar,Dim>,Dim> &particle_domain_corner) const
{
    if(object_idx >= this->objectNum())
    {
        std::cerr<<"Warning: object index out of range, return empty vector!\n";
        particle_domain_corner.clear();
    }
    else if(particle_idx >= this->particleNumOfObject(object_idx))
    {
        std::cerr<<"Warning: particle index out of range, return empty vector!\n";
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
            *iter = particle_domain_corners_[object_idx][particle_idx][idx_1d];
        }
    }
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::setCurrentParticleDomain(unsigned int object_idx, unsigned int particle_idx,
                                                                const ArrayND<Vector<Scalar,Dim>,Dim> &particle_domain_corner)
{
    if(object_idx >= this->objectNum())
        std::cerr<<"Warning: object index out of range, operation ignored!\n";
    else if(particle_idx >= this->particleNumOfObject(object_idx))
        std::cerr<<"Warning: particle index out of range, operation ignored!\n";
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
                particle_domain_corners_[object_idx][particle_idx][idx_1d] = *iter;
            }
        }
        else
            std::cerr<<"Warning: invalid number of domain corners provided, operation ignored!\n";
    }
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::initialParticleDomain(unsigned int object_idx, unsigned int particle_idx,
                                                             ArrayND<Vector<Scalar,Dim>,Dim> &particle_domain_corner) const
{
    if(object_idx >= this->objectNum())
    {
        std::cerr<<"Warning: object index out of range, return empty vector!\n";
        particle_domain_corner.clear();
    }
    else if(particle_idx >= this->particleNumOfObject(object_idx))
    {
        std::cerr<<"Warning: particle index out of range, return empty vector!\n";
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
            *iter = initial_particle_domain_corners_[object_idx][particle_idx][idx_1d];
        }
    }
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::initializeParticleDomain(unsigned int object_idx, unsigned int particle_idx,
                                                                const ArrayND<Vector<Scalar,Dim>,Dim> &particle_domain_corner)
{
    if(object_idx >= this->objectNum())
        std::cerr<<"Warning: object index out of range, operation ignored!\n";
    else if(particle_idx >= this->particleNumOfObject(object_idx))
        std::cerr<<"Warning: particle index out of range, operation ignored!\n";
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
                particle_domain_corners_[object_idx][particle_idx][idx_1d] = *iter;
                initial_particle_domain_corners_[object_idx][particle_idx][idx_1d] = *iter;
            }
        }
        else
            std::cerr<<"Warning: invalid number of domain corners provided, operation ignored!\n";
    }
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> CPDIMPMSolid<Scalar,Dim>::currentParticleDomainCorner(unsigned int object_idx, unsigned int particle_idx,
                                                                                 const Vector<unsigned int,Dim> &corner_idx) const
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("object index out of range!");
    else if(particle_idx >= this->particleNumOfObject(object_idx))
        throw PhysikaException("particle index out of range!");
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
        throw PhysikaException("corner index out of range!");
    return particle_domain_corners_[object_idx][particle_idx][idx_1d];
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> CPDIMPMSolid<Scalar,Dim>::initialParticleDomainCorner(unsigned int object_idx, unsigned int particle_idx,
                                                                                 const Vector<unsigned int,Dim> &corner_idx) const
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("object index out of range!");
    else if(particle_idx >= this->particleNumOfObject(object_idx))
        throw PhysikaException("particle index out of range!");
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
        throw PhysikaException("corner index out of range!");
    return initial_particle_domain_corners_[object_idx][particle_idx][idx_1d];
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::synchronizeWithInfluenceRangeChange()
{
    //for each particle domain corner, allocate space that can store weight/gradient of maximum
    //number of nodes in range of the domain corners
    unsigned int max_num = 1;
    for(unsigned int i = 0; i < Dim; ++i)
        max_num *= static_cast<unsigned int>((this->weight_function_->supportRadius())*2+1);
    unsigned int corner_num = Dim==2 ? 4 : 8;
    for(unsigned int i = 0; i < this->objectNum(); ++i)
        for(unsigned int j = 0; j < this->particleNumOfObject(i); ++j)
        {
            for(unsigned int k = 0; k < corner_num; ++k)
                corner_grid_weight_[i][j][k].resize(max_num);
            //the maximum number of nodes in range of particles is the sum of domain corners
            this->particle_grid_weight_and_gradient_[i][j].resize(max_num*corner_num);
        }
}

template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::appendAllParticleRelatedDataOfLastObject()
{
    MPMSolidBase<Scalar,Dim>::appendAllParticleRelatedDataOfLastObject();
    unsigned int corner_num = Dim==2 ? 4 : 8;
    unsigned int last_object_idx = this->objectNum() - 1;
    unsigned int particle_num_of_last_object = this->particleNumOfObject(last_object_idx);
    std::vector<Vector<Scalar,Dim> > particle_domain_corners(corner_num);
    std::vector<std::vector<Vector<Scalar,Dim> > > all_particle_domain_corners(particle_num_of_last_object,particle_domain_corners);
    for(unsigned int i = 0; i < particle_num_of_last_object; ++i)
    {
        const SolidParticle<Scalar,Dim> &particle = this->particle(last_object_idx,i);
        initParticleDomain(particle,all_particle_domain_corners[i]);
    }
    particle_domain_corners_.push_back(all_particle_domain_corners);
    initial_particle_domain_corners_.push_back(all_particle_domain_corners);
    unsigned int max_num = 1;
    for(unsigned int i = 0; i < Dim; ++i)
        max_num *= static_cast<unsigned int>((this->weight_function_->supportRadius())*2+1);
    typedef MPMInternal::NodeIndexWeightPair<Scalar,Dim> weight_pair;
    std::vector<weight_pair> one_corner_grid_weight(max_num);
    std::vector<std::vector<weight_pair> > all_corner_grid_weight(corner_num,one_corner_grid_weight);
    std::vector<std::vector<std::vector<weight_pair> > > all_particle_corner_grid_weight(particle_num_of_last_object,all_corner_grid_weight);
    corner_grid_weight_.push_back(all_particle_corner_grid_weight);
    std::vector<unsigned int> all_corner_grid_pair_num(corner_num);
    std::vector<std::vector<unsigned int> > all_particle_corner_grid_pair_num(particle_num_of_last_object,all_corner_grid_pair_num);
    corner_grid_pair_num_.push_back(all_particle_corner_grid_pair_num);
    //the maximum number of nodes in range of particles is the sum of domain corners
    typedef MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> weight_gradient_pair;
    std::vector<weight_gradient_pair> one_particle_grid_weight_and_gradient(max_num*corner_num);
    std::vector<std::vector<weight_gradient_pair> > all_particle_grid_weight_and_gradient(particle_num_of_last_object,one_particle_grid_weight_and_gradient);
    this->particle_grid_weight_and_gradient_.push_back(all_particle_grid_weight_and_gradient);
    std::vector<unsigned int> all_particle_grid_pair_num(particle_num_of_last_object);
    this->particle_grid_pair_num_.push_back(all_particle_grid_pair_num);
}
    
template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::appendLastParticleRelatedDataOfObject(unsigned int object_idx)
{
    MPMSolidBase<Scalar,Dim>::appendLastParticleRelatedDataOfObject(object_idx);
    unsigned int last_particle_idx = this->particleNumOfObject(object_idx) - 1;
    const SolidParticle<Scalar,Dim> &last_particle = this->particle(object_idx,last_particle_idx);
    unsigned int corner_num = Dim==2 ? 4 : 8;
    std::vector<Vector<Scalar,Dim> > particle_domain_corners(corner_num);
    initParticleDomain(last_particle,particle_domain_corners);
    particle_domain_corners_[object_idx].push_back(particle_domain_corners);
    initial_particle_domain_corners_[object_idx].push_back(particle_domain_corners);
    unsigned int max_num = 1;
    for(unsigned int i = 0; i < Dim; ++i)
        max_num *= static_cast<unsigned int>((this->weight_function_->supportRadius())*2+1);
    typedef MPMInternal::NodeIndexWeightPair<Scalar,Dim> weight_pair;
    std::vector<weight_pair> one_corner_grid_weight(max_num);
    std::vector<std::vector<weight_pair> > all_corner_grid_weight(corner_num,one_corner_grid_weight);
    corner_grid_weight_[object_idx].push_back(all_corner_grid_weight);
    std::vector<unsigned int> all_corner_grid_pair_num(corner_num);
    corner_grid_pair_num_[object_idx].push_back(all_corner_grid_pair_num);
    //the maximum number of nodes in range of particles is the sum of domain corners
    typedef MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim>  weight_gradient_pair;
    std::vector<weight_gradient_pair> one_particle_grid_weight_and_gradient(max_num*corner_num);
    this->particle_grid_weight_and_gradient_[object_idx].push_back(one_particle_grid_weight_and_gradient);
    this->particle_grid_pair_num_[object_idx].push_back(0);
}
    
template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::deleteAllParticleRelatedDataOfObject(unsigned int object_idx)
{
    MPMSolid<Scalar,Dim>::deleteAllParticleRelatedDataOfObject(object_idx);
    typename std::vector<std::vector<std::vector<Vector<Scalar,Dim> > > >::iterator iter1
        = particle_domain_corners_.begin() + object_idx;
    particle_domain_corners_.erase(iter1);
    typename std::vector<std::vector<std::vector<Vector<Scalar,Dim> > > >::iterator iter2
        = initial_particle_domain_corners_.begin() + object_idx;
    initial_particle_domain_corners_.erase(iter2);
    typename std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,Dim> > > > >::iterator
        iter3 = corner_grid_weight_.begin() + object_idx;
    corner_grid_weight_.erase(iter3);
    typename std::vector<std::vector<std::vector<unsigned int> > >::iterator iter4 = corner_grid_pair_num_.begin() + object_idx;
    corner_grid_pair_num_.erase(iter4);
}
 
template <typename Scalar, int Dim>
void CPDIMPMSolid<Scalar,Dim>::deleteOneParticleRelatedDataOfObject(unsigned int object_idx, unsigned int particle_idx)
{
    MPMSolid<Scalar,Dim>::deleteOneParticleRelatedDataOfObject(object_idx,particle_idx);
    typename std::vector<std::vector<Vector<Scalar,Dim> > >::iterator iter1 = particle_domain_corners_[object_idx].begin() + particle_idx;
    particle_domain_corners_[object_idx].erase(iter1);
    typename std::vector<std::vector<Vector<Scalar,Dim> > >::iterator iter2 = initial_particle_domain_corners_[object_idx].begin() + particle_idx;
    initial_particle_domain_corners_[object_idx].erase(iter2);
    typename std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,Dim> > > >::iterator iter3 =
        corner_grid_weight_[object_idx].begin() + particle_idx;
    corner_grid_weight_[object_idx].erase(iter3);
    typename std::vector<std::vector<unsigned int> >::iterator iter4 = corner_grid_pair_num_[object_idx].begin() + particle_idx;
    corner_grid_pair_num_[object_idx].erase(iter4);
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
    Scalar particle_radius = pow(particle.volume(),static_cast<Scalar>(1.0/3.0))/2.0;//assume the particle occupies cubic space
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

//explicit instantiations
template class CPDIMPMSolid<float,2>;
template class CPDIMPMSolid<float,3>;
template class CPDIMPMSolid<double,2>;
template class CPDIMPMSolid<double,3>;

}  //end of namespace Physika
