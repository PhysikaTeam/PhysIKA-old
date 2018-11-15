/*
* @file enriched_mpm_uniform_grid_generalized_vector.cpp
* @brief generalized vector for mpm drivers with uniform grid && enriched DOFs
*        defined for element type Vector<Scalar,Dim>
* @author Fei Zhu
*
* This file is part of Physika, a versatile physics simulation library.
* Copyright (C) 2013- Physika Group.
*
* This Source Code Form is subject to the terms of the GNU General Public License v2.0.
* If a copy of the GPL was not distributed with this file, you can obtain one at:
* http://www.gnu.org/licenses/gpl-2.0.html
*
*/

#include <typeinfo>
#include <set>
#include <algorithm>
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Geometry/Volumetric_Meshes/quad_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/cubic_mesh.h"
#include "Physika_Dynamics/MPM/MPM_Linear_Systems/enriched_mpm_uniform_grid_generalized_vector.h"

namespace Physika{

template <typename Scalar, int Dim>
EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >::EnrichedMPMUniformGridGeneralizedVector
                                                             (const Vector<unsigned int, Dim> &grid_size)
                                                              :grid_data_(grid_size)
{
    domain_corner_data_.clear();
    enriched_domain_corners_.clear();
    particle_domain_topology_.clear();
}

template <typename Scalar, int Dim>
EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >::EnrichedMPMUniformGridGeneralizedVector
                                      (const Vector<unsigned int, Dim> &grid_size,
                                       const std::vector<Vector<unsigned, Dim> > &active_grid_nodes,
                                       const std::vector<std::vector<unsigned int> > &enriched_particles,
                                       const std::vector<VolumetricMesh<Scalar,Dim>*> &particle_domain_topology)
                                       :grid_data_(grid_size,active_grid_nodes)
{
    if (enriched_particles.size() != particle_domain_topology.size())
        throw PhysikaException("Inconsistent object number: provided vector size mismatch!");
    unsigned int obj_num = enriched_particles.size();
    domain_corner_data_.resize(obj_num);
    enriched_domain_corners_.resize(obj_num);
    for (unsigned int obj_idx = 0; obj_idx < obj_num; ++obj_idx)
    {
        PHYSIKA_ASSERT(particle_domain_topology[obj_idx]);
        domain_corner_data_[obj_idx].resize(particle_domain_topology[obj_idx]->vertNum(), Vector<Scalar, Dim>(0));
        setEnrichedDomainCorners(enriched_particles[obj_idx], particle_domain_topology[obj_idx], enriched_domain_corners_[obj_idx]);
    }
    particle_domain_topology_ = particle_domain_topology;
}

template <typename Scalar, int Dim>
EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::EnrichedMPMUniformGridGeneralizedVector(
                          const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &vector)
                          :grid_data_(vector.grid_data_), domain_corner_data_(vector.domain_corner_data_),
                          enriched_domain_corners_(vector.enriched_domain_corners_), particle_domain_topology_(vector.particle_domain_topology_)
{

}

template <typename Scalar, int Dim>
EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::~EnrichedMPMUniformGridGeneralizedVector()
{

}

template <typename Scalar, int Dim>
EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >& EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::
   operator= (const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &vector)
{
    grid_data_ = vector.grid_data_;
    domain_corner_data_ = vector.domain_corner_data_;
    enriched_domain_corners_ = vector.enriched_domain_corners_;
    particle_domain_topology_ = vector.particle_domain_topology_;
    return *this;
}

template <typename Scalar, int Dim>
EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >* EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::clone() const
{
    return new EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >(*this);
}

template <typename Scalar, int Dim>
unsigned int EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::size() const
{
    unsigned int enriched_corner_num = 0;
    for (unsigned int i = 0; i < enriched_domain_corners_.size(); ++i)
        enriched_corner_num += enriched_domain_corners_[i].size();
    return grid_data_.size() + enriched_corner_num;
}

template <typename Scalar, int Dim>
EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >& EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >
                                                              ::operator+= (const GeneralizedVector<Scalar> &vector)
{
    try{
        const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &enrich_vec =
         dynamic_cast<const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &>(vector);
        bool same_pattern = checkActivePattern(enrich_vec);
        if(!same_pattern)
            throw PhysikaException("Active entry pattern mismatch!");
        grid_data_ += enrich_vec.grid_data_;
        for (unsigned int idx = 0; idx < enriched_domain_corners_.size(); ++idx)
        {
            for (unsigned int c_idx = 0; c_idx < enriched_domain_corners_[idx].size(); ++c_idx)
            {
                unsigned int enriched_corner_idx = enriched_domain_corners_[idx][c_idx];
                domain_corner_data_[idx][enriched_corner_idx] += enrich_vec.domain_corner_data_[idx][enriched_corner_idx];
            }
        }
    }
    catch(std::bad_cast &e)
    {
     throw PhysikaException("Incorrect argument!");
    }
    return *this;
}

template <typename Scalar, int Dim>
EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >& EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >
                                                               ::operator-= (const GeneralizedVector<Scalar> &vector)
{
    try{
        const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &enrich_vec =
         dynamic_cast<const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &>(vector);
        bool same_pattern = checkActivePattern(enrich_vec);
        if(!same_pattern)
            throw PhysikaException("Active entry pattern mismatch!");
        grid_data_ -= enrich_vec.grid_data_;
        for (unsigned int idx = 0; idx < enriched_domain_corners_.size(); ++idx)
        {
            for (unsigned int c_idx = 0; c_idx < enriched_domain_corners_[idx].size(); ++c_idx)
            {
                unsigned int enriched_corner_idx = enriched_domain_corners_[idx][c_idx];
                domain_corner_data_[idx][enriched_corner_idx] -= enrich_vec.domain_corner_data_[idx][enriched_corner_idx];
            }
        }
    }
    catch(std::bad_cast &e)
    {
     throw PhysikaException("Incorrect argument!");
    }
    return *this;
}

template <typename Scalar, int Dim>
EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >& EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >
                                                              ::operator*= (Scalar value)
{
    grid_data_ *= value;
    for (unsigned int idx = 0; idx < enriched_domain_corners_.size(); ++idx)
    {
        for (unsigned int c_idx = 0; c_idx < enriched_domain_corners_[idx].size(); ++c_idx)
        {
            unsigned int enriched_corner_idx = enriched_domain_corners_[idx][c_idx];
            domain_corner_data_[idx][enriched_corner_idx] *= value;
        }
    }
    return *this;
}

template <typename Scalar, int Dim>
EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >& EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >
                                                               ::operator/= (Scalar value)
{
    //divide by zero will be checked by element type
    grid_data_ /= value;
    for (unsigned int idx = 0; idx < enriched_domain_corners_.size(); ++idx)
    {
        for (unsigned int c_idx = 0; c_idx < enriched_domain_corners_[idx].size(); ++c_idx)
        {
            unsigned int enriched_corner_idx = enriched_domain_corners_[idx][c_idx];
            domain_corner_data_[idx][enriched_corner_idx] /= value;
        }
    }
    return *this;
}

template <typename Scalar, int Dim>
const Vector<Scalar,Dim>& EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::operator[] (const Vector<unsigned int, Dim> &idx) const
{
    return grid_data_[idx];
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim>& EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::operator[] (const Vector<unsigned int, Dim> &idx)
{
    return grid_data_[idx];
}


template <typename Scalar, int Dim>
const Vector<Scalar,Dim>& EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::operator() (unsigned int object_idx, unsigned int particle_idx, unsigned int corner_idx) const
{
    if (object_idx >= domain_corner_data_.size())
        throw PhysikaException("object index out of range!");
    if (particle_idx >= particle_domain_topology_[object_idx]->eleNum())
        throw PhysikaException("particle index out of range!");
    unsigned int corner_num = Dim == 2 ? 4 : 8;
    if (corner_idx >= corner_num)
        throw PhysikaException("corner index out of range!");
    unsigned int global_corner_idx = particle_domain_topology_[object_idx]->eleVertIndex(particle_idx, corner_idx);
    return domain_corner_data_[object_idx][global_corner_idx];
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim>& EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::operator() (unsigned int object_idx, unsigned int particle_idx, unsigned int corner_idx)
{
    if (object_idx >= domain_corner_data_.size())
        throw PhysikaException("object index out of range!");
    if (particle_idx >= particle_domain_topology_[object_idx]->eleNum())
        throw PhysikaException("particle index out of range!");
    unsigned int corner_num = Dim == 2 ? 4 : 8;
    if (corner_idx >= corner_num)
        throw PhysikaException("corner index out of range!");
    unsigned int global_corner_idx = particle_domain_topology_[object_idx]->eleVertIndex(particle_idx, corner_idx);
    return domain_corner_data_[object_idx][global_corner_idx];
}

template <typename Scalar, int Dim>
void EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::setValue(const Vector<Scalar, Dim> &value)
{
    grid_data_.setValue(value);
    for (unsigned int i = 0; i < enriched_domain_corners_.size(); ++i)
        for (unsigned int j = 0; j < enriched_domain_corners_[i].size(); ++j)
        {
            unsigned int idx = enriched_domain_corners_[i][j];
            domain_corner_data_[i][idx] = value;
        }
}

template <typename Scalar, int Dim>
void EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::setActivePattern(const std::vector<Vector<unsigned int, Dim> > &active_grid_nodes,
                                                                                     const std::vector<std::vector<unsigned int> > &enriched_particles,
                                                                                     const std::vector<VolumetricMesh<Scalar,Dim>*> &particle_domain_topology)
{
    if (enriched_particles.size() != particle_domain_topology.size())
        throw PhysikaException("Inconsistent object number: provided vector size mismatch!");
    grid_data_.setActivePattern(active_grid_nodes);
    unsigned int obj_num = enriched_particles.size();
    domain_corner_data_.resize(obj_num);
    enriched_domain_corners_.resize(obj_num);
    for (unsigned int obj_idx = 0; obj_idx < obj_num; ++obj_idx)
    {
        PHYSIKA_ASSERT(particle_domain_topology[obj_idx]);
        domain_corner_data_[obj_idx].resize(particle_domain_topology[obj_idx]->vertNum(), Vector<Scalar, Dim>(0));
        setEnrichedDomainCorners(enriched_particles[obj_idx], particle_domain_topology[obj_idx], enriched_domain_corners_[obj_idx]);
    }
    particle_domain_topology_ = particle_domain_topology;    
}

template <typename Scalar, int Dim>
void EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::copy(const GeneralizedVector<Scalar> &vector)
{
    try{
        const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &mpm_vec = dynamic_cast<const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >&>(vector);
        this->grid_data_ = mpm_vec.grid_data_;
        this->domain_corner_data_ = mpm_vec.domain_corner_data_;
        this->enriched_domain_corners_ = mpm_vec.enriched_domain_corners_;
        this->particle_domain_topology_ = mpm_vec.particle_domain_topology_;    }
    catch (std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
}

template <typename Scalar, int Dim>
bool EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::checkActivePattern(const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &vector) const
{
    if (grid_data_.checkActivePattern(vector.grid_data_) == false)
        return false;
    if (enriched_domain_corners_.size() != vector.enriched_domain_corners_.size())
        return false;
    for (unsigned int i = 0; i < enriched_domain_corners_.size(); ++i)
        if (enriched_domain_corners_[i] != vector.enriched_domain_corners_[i])
            return false;
    return true;
}

template <typename Scalar, int Dim>
void EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::setEnrichedDomainCorners(const std::vector<unsigned int> &enriched_particles,
                                                                                             const VolumetricMesh<Scalar,Dim> *particle_domain_topology,
                                                                                             std::vector<unsigned int> &enriched_domain_corners)
{
    PHYSIKA_ASSERT(particle_domain_topology);
    enriched_domain_corners.clear();
    std::set<unsigned int> corner_set;
    for (unsigned int i = 0; i < enriched_particles.size(); ++i)
    {
        unsigned int particle_idx = enriched_particles[i];
        for (unsigned int j = 0; j < particle_domain_topology->eleVertNum(particle_idx); ++j)
            corner_set.insert(particle_domain_topology->eleVertIndex(particle_idx, j));
    }
    for (std::set<unsigned int>::iterator iter = corner_set.begin(); iter != corner_set.end(); ++iter)
        enriched_domain_corners.push_back(*iter);
    //sort in ascending order
    std::sort(enriched_domain_corners.begin(), enriched_domain_corners.end());
}

//explicit instantiations
template class EnrichedMPMUniformGridGeneralizedVector<Vector<float,2> >;
template class EnrichedMPMUniformGridGeneralizedVector<Vector<float,3> >;
template class EnrichedMPMUniformGridGeneralizedVector<Vector<double,2> >;
template class EnrichedMPMUniformGridGeneralizedVector<Vector<double,3> >;

}  //end of namespace Physika
