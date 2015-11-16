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
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Geometry/Volumetric_Meshes/quad_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/cubic_mesh.h"
#include "Physika_Dynamics/MPM/MPM_Linear_Systems/enriched_mpm_uniform_grid_generalized_vector.h"

namespace Physika{

template <typename Scalar, int Dim>
EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >::EnrichedMPMUniformGridGeneralizedVector
                                                             (const Vector<unsigned int, Dim> &grid_size,
                                                              unsigned int object_num,
                                                              const std::vector<unsigned int> &particle_num,
                                                              const std::vector<VolumetricMesh<Scalar,Dim>*> &particle_domain_topology)
                                                              :grid_data_(grid_size)
{
    domain_corner_data_.resize(object_num);
    if (particle_num.size() != object_num)
        throw PhysikaException("object_num and particle_num vector size mismatch!");
    if (particle_domain_topology.size() != object_num)
        throw PhysikaException("object_num and particle_domain_topology vector size mismatch!");
    for (unsigned int obj_idx = 0; obj_idx < object_num; ++obj_idx)
    {
    }
   //no domain corner enriched
    enriched_particles_.clear();
    particle_domain_topology_ = particle_domain_topology;
}

template <typename Scalar, int Dim>
EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >::EnrichedMPMUniformGridGeneralizedVector
                                      (const Vector<unsigned int, Dim> &grid_size,
                                       unsigned int object_num,
                                       const std::vector<unsigned int> &particle_num,
                                       const std::vector<Vector<unsigned, Dim> > &active_grid_nodes,
                                       const std::vector<std::vector<unsigned int> > &enriched_particles,
                                       const std::vector<VolumetricMesh<Scalar,Dim>*> &particle_domain_topology)
                                       :grid_data_(grid_size,active_grid_nodes)
{
    unsigned int corner_num = Dim == 2 ? 4 : 8;
    domain_corner_data_.resize(object_num);
    if (particle_num.size() != object_num)
        throw PhysikaException("object_num and particle_num vector size mismatch!");
    for (unsigned int obj_idx = 0; obj_idx < object_num; ++obj_idx)
    {
        domain_corner_data_[obj_idx].resize(particle_num[obj_idx]);
        for (unsigned int particle_idx = 0; particle_idx < particle_num[obj_idx]; ++particle_idx)
            domain_corner_data_[obj_idx][particle_idx].resize(corner_num, Vector<Scalar, Dim>(0));
    }
   active_particle_domain_corners_ = active_domain_corners;
   sortActiveDomainCorners();
}

template <typename Scalar, int Dim>
EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::EnrichedMPMUniformGridGeneralizedVector(
                          const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &vector)
                          :grid_data_(vector.grid_data_), domain_corner_data_(vector.domain_corner_data_),
                           active_particle_domain_corners_(vector.active_particle_domain_corners_)
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
    active_particle_domain_corners_ = vector.active_particle_domain_corners_;
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
    return grid_data_.size() + active_particle_domain_corners_.size();
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
        for(unsigned int active_idx = 0; active_idx < active_particle_domain_corners_.size(); ++active_idx)
        {
            unsigned int object_idx = active_particle_domain_corners_[active_idx][0];
            unsigned int particle_idx = active_particle_domain_corners_[active_idx][1];
            unsigned int corner_idx = active_particle_domain_corners_[active_idx][2];
            domain_corner_data_[object_idx][particle_idx][corner_idx] += enrich_vec.domain_corner_data_[object_idx][particle_idx][corner_idx];
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
        for (unsigned int active_idx = 0; active_idx < active_particle_domain_corners_.size(); ++active_idx)
        {
            unsigned int object_idx = active_particle_domain_corners_[active_idx][0];
            unsigned int particle_idx = active_particle_domain_corners_[active_idx][1];
            unsigned int corner_idx = active_particle_domain_corners_[active_idx][2];
            domain_corner_data_[object_idx][particle_idx][corner_idx] -= enrich_vec.domain_corner_data_[object_idx][particle_idx][corner_idx];
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
    for(unsigned int active_idx = 0; active_idx < active_particle_domain_corners_.size(); ++active_idx)
    {
        unsigned int object_idx = active_particle_domain_corners_[active_idx][0];
        unsigned int particle_idx = active_particle_domain_corners_[active_idx][1];
        unsigned int corner_idx = active_particle_domain_corners_[active_idx][2];
        domain_corner_data_[object_idx][particle_idx][corner_idx] *= value;
    }
    return *this;
}

template <typename Scalar, int Dim>
EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >& EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >
                                                               ::operator/= (Scalar value)
{
    //divide by zero will be checked by element type
    grid_data_ /= value;
    for(unsigned int active_idx = 0; active_idx < active_particle_domain_corners_.size(); ++active_idx)
    {
        unsigned int object_idx = active_particle_domain_corners_[active_idx][0];
        unsigned int particle_idx = active_particle_domain_corners_[active_idx][1];
        unsigned int corner_idx = active_particle_domain_corners_[active_idx][2];
        domain_corner_data_[object_idx][particle_idx][corner_idx] /= value;
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
    if (object_idx >= this->domain_corner_data_.size())
        throw PhysikaException("Object index out of range!");
    if (particle_idx >= this->domain_corner_data_[object_idx].size())
        throw PhysikaException("Particle index out of range!");
    if (corner_idx >= this->domain_corner_data_[object_idx][particle_idx].size())
        throw PhysikaException("Corner index out of range!");
    return domain_corner_data_[object_idx][particle_idx][corner_idx];
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim>& EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::operator() (unsigned int object_idx, unsigned int particle_idx, unsigned int corner_idx)
{
    if (object_idx >= this->domain_corner_data_.size())
        throw PhysikaException("Object index out of range!");
    if (particle_idx >= this->domain_corner_data_[object_idx].size())
        throw PhysikaException("Particle index out of range!");
    if (corner_idx >= this->domain_corner_data_[object_idx][particle_idx].size())
        throw PhysikaException("Corner index out of range!");
    return domain_corner_data_[object_idx][particle_idx][corner_idx];
}

template <typename Scalar, int Dim>
void EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::setValue(const Vector<Scalar, Dim> &value)
{
    grid_data_.setValue(value);
    for (unsigned int i = 0; i < this->active_particle_domain_corners_.size(); ++i)
    {
        unsigned int obj_idx = this->active_particle_domain_corners_[i][0];
        unsigned int particle_idx = this->active_particle_domain_corners_[i][1];
        unsigned int corner_idx = this->active_particle_domain_corners_[i][2];
        domain_corner_data_[obj_idx][particle_idx][corner_idx] = value;
    }
}

template <typename Scalar, int Dim>
void EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::setActivePattern(const std::vector<Vector<unsigned int, Dim> > &active_grid_nodes,
                                                                                     const std::vector<Vector<unsigned int> > &enriched_particles)
{
  //TO DO
}

template <typename Scalar, int Dim>
void EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::copy(const GeneralizedVector<Scalar> &vector)
{
    try{
        const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &mpm_vec = dynamic_cast<const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >&>(vector);
        this->grid_data_ = mpm_vec.grid_data_;
        this->domain_corner_data_ = mpm_vec.domain_corner_data_;
        this->active_particle_domain_corners_ = mpm_vec.active_particle_domain_corners_;
    }
    catch (std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
}

template <typename Scalar, int Dim>
bool EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::checkActivePattern(const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &vector) const
{
    
    return false;
}

template <typename Scalar, int Dim>
void EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::sortActiveDomainCorners()
{
  //TO DO
}

//explicit instantiations
template class EnrichedMPMUniformGridGeneralizedVector<Vector<float,2> >;
template class EnrichedMPMUniformGridGeneralizedVector<Vector<float,3> >;
template class EnrichedMPMUniformGridGeneralizedVector<Vector<double,2> >;
template class EnrichedMPMUniformGridGeneralizedVector<Vector<double,3> >;

}  //end of namespace Physika
