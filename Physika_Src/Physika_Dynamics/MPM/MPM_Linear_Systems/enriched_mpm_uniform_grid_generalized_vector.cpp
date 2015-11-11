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
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Dynamics/MPM/MPM_Linear_Systems/enriched_mpm_uniform_grid_generalized_vector.h"

namespace Physika{

//template <typename Scalar, int Dim>
//EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >::EnrichedMPMUniformGridGeneralizedVector
//                                                             (const Vector<unsigned int, Dim> &grid_size,
//                                                              unsigned int particle_num)
//                                                              :grid_data_(grid_size)
//{
//   domain_corner_data_.resize(particle_num);
//   unsigned int corner_num = Dim == 2 ? 4 : 8;
//   for (unsigned int i = 0; i < particle_num; ++i)
//       domain_corner_data_[i].resize(corner_num, Vector<Scalar, Dim>(0));
//   //no domain corner enriched
//   active_particle_domain_corners_.clear();
//   active_domain_corner_mass_.clear();
//}
//
//template <typename Scalar, int Dim>
//EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >::EnrichedMPMUniformGridGeneralizedVector
//                                      (const Vector<unsigned int, Dim> &grid_size,
//                                       unsigned int particle_num,
//                                       const std::vector<Vector<unsigned, Dim> > &active_grid_nodes,
//                                       const std::vector<Scalar> &active_node_mass,
//                                       const std::vector<Vector<unsigned int, 2> >&active_domain_corners,
//                                       const std::vector<Scalar> &active_domain_corner_mass)
//                                       :grid_data_(grid_size,active_grid_nodes,active_node_mass)
//{
//   domain_corner_data_.resize(particle_num);
//   unsigned int corner_num = Dim == 2 ? 4 : 8;
//   for (unsigned int i = 0; i < particle_num; ++i)
//       domain_corner_data_[i].resize(corner_num, Vector<Scalar, Dim>(0));
//   active_particle_domain_corners_ = active_domain_corners;
//   sortActiveDomainCorners();
//   active_domain_corner_mass_ = active_domain_corner_mass;
//}
//
//template <typename Scalar, int Dim>
//EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::EnrichedMPMUniformGridGeneralizedVector(
//                          const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &vector)
//                          :grid_data_(vector.grid_data_), domain_corner_data_(vector.domain_corner_data_),
//                           active_particle_domain_corners_(vector.active_particle_domain_corners_),
//                           active_domain_corner_mass_(vector.active_domain_corner_mass_)
//{
//
//}
//
//template <typename Scalar, int Dim>
//EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::~EnrichedMPMUniformGridGeneralizedVector()
//{
//
//}
//
//template <typename Scalar, int Dim>
//EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >& EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::
//   operator= (const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &vector)
//{
//    grid_data_ = vector.grid_data_;
//    domain_corner_data_ = vector.domain_corner_data_;
//    active_particle_domain_corners_ = vector.active_particle_domain_corners_;
//    active_domain_corner_mass_ = vector.active_domain_corner_mass_;
//    return *this;
//}
//
//template <typename Scalar, int Dim>
//EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >* EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::clone() const
//{
//    return new EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >(*this);
//}
//
//template <typename Scalar, int Dim>
//unsigned int EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::size() const
//{
//    return grid_data_.size() + active_particle_domain_corners_.size();
//}
//
//template <typename Scalar, int Dim>
//EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >& EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >
//                                                              ::operator+= (const GeneralizedVector<Scalar> &vector)
//{
//    try{
//        const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &enrich_vec =
//         dynamic_cast<const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &>(vector);
//        bool same_pattern = checkActivePattern(enrich_vec);
//        if(!same_pattern)
//            throw PhysikaException("Active entry pattern mismatch!");
//        grid_data_ += enrich_vec.grid_data_;
//        for(unsigned int active_idx = 0; active_idx < active_particle_domain_corners_.size(); ++active_idx)
//        {
//            unsigned int particle_idx = active_particle_domain_corners_[active_idx][0];
//            unsigned int corner_idx = active_particle_domain_corners_[active_idx][1];
//            domain_corner_data_[particle_idx][corner_idx] += enrich_vec.domain_corner_data_[particle_idx][corner_idx];
//        }
//    }
//    catch(std::bad_cast &e)
//    {
//     throw PhysikaException("Incorrect argument!");
//    }
//    return *this;
//}
//
//template <typename Scalar, int Dim>
//EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >& EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >
//                                                               ::operator-= (const GeneralizedVector<Scalar> &vector)
//{
//    try{
//        const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &enrich_vec =
//         dynamic_cast<const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &>(vector);
//        bool same_pattern = checkActivePattern(enrich_vec);
//        if(!same_pattern)
//            throw PhysikaException("Active entry pattern mismatch!");
//        grid_data_ -= enrich_vec.grid_data_;
//        for(unsigned int active_idx = 0; active_idx < active_particle_domain_corners_.size(); ++active_idx)
//        {
//            unsigned int particle_idx = active_particle_domain_corners_[active_idx][0];
//            unsigned int corner_idx = active_particle_domain_corners_[active_idx][1];
//            domain_corner_data_[particle_idx][corner_idx] -= enrich_vec.domain_corner_data_[particle_idx][corner_idx];
//        }
//    }
//    catch(std::bad_cast &e)
//    {
//     throw PhysikaException("Incorrect argument!");
//    }
//    return *this;
//}
//
//template <typename Scalar, int Dim>
//EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >& EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >
//                                                              ::operator*= (Scalar value)
//{
//    grid_data_ *= value;
//    for(unsigned int active_idx = 0; active_idx < active_particle_domain_corners_.size(); ++active_idx)
//    {
//        unsigned int particle_idx = active_particle_domain_corners_[active_idx][0];
//        unsigned int corner_idx = active_particle_domain_corners_[active_idx][1];
//        domain_corner_data_[particle_idx][corner_idx] *= value;
//    }
//    return *this;
//}
//
//template <typename Scalar, int Dim>
//EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >& EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >
//                                                               ::operator/= (Scalar value)
//{
//    //divide by zero will be checked by element type
//    grid_data_ /= value;
//    for(unsigned int active_idx = 0; active_idx < active_particle_domain_corners_.size(); ++active_idx)
//    {
//        unsigned int particle_idx = active_particle_domain_corners_[active_idx][0];
//        unsigned int corner_idx = active_particle_domain_corners_[active_idx][1];
//        domain_corner_data_[particle_idx][corner_idx] /= value;
//    }
//    return *this;
//}
//
//template <typename Scalar, int Dim>
//Scalar EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::norm() const
//{
//    return sqrt(normSquared());
//}
//
//template <typename Scalar, int Dim>
//Scalar EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::normSquared() const
//{
//    //TO DO
//    return 0;
//}
//
//template <typename Scalar, int Dim>
//Scalar EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::dot(const GeneralizedVector<Scalar> &vector) const
//{
//    //TO DO
//    return 0;
//}
//
//template <typename Scalar, int Dim>
//const Vector<Scalar,Dim>& EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::operator() (const Vector<unsigned int, Dim> &idx) const
//{
//  //TO DO
//}
//
//template <typename Scalar, int Dim>
//Vector<Scalar,Dim>& EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::operator() (const Vector<unsigned int, Dim> &idx)
//{
//  //TO DO
//}
//
//
//template <typename Scalar, int Dim>
//const Vector<Scalar,Dim>& EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::operator() (unsigned int particle_idx, unsigned int corner_idx) const
//{
//  //TO DO
//}
//
//template <typename Scalar, int Dim>
//Vector<Scalar,Dim>& EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::operator() (unsigned int particle_idx, unsigned int corner_idx)
//{
//  //TO DO
//}
//
//template <typename Scalar, int Dim>
//void EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::setValue(const Vector<Scalar, Dim> &value)
//{
//  //TO DO
//}
//
//template <typename Scalar, int Dim>
//void EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::setActivePattern(const std::vector<Vector<unsigned int, Dim> > &active_grid_nodes,
//                                                                                     const std::vector<Vector<unsigned int, 2> > &active_particle_domain_corners)
//{
//  //TO DO
//}
//
//template <typename Scalar, int Dim>
//void EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::setActiveNodeMass(const std::vector<Scalar> &active_node_mass)
//{
//  //TO DO
//}
//
//template <typename Scalar, int Dim>
//void EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::setActiveDomainCornerMass(const std::vector<Scalar> &active_corner_mass)
//{
//  //TO DO
//}
//
//template <typename Scalar, int Dim>
//void EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::copy(const GeneralizedVector<Scalar> &vector)
//{
//  //TO DO
//}
//
//template <typename Scalar, int Dim>
//bool EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::checkActivePattern(const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &vector) const
//{
//    //TO DO
//    return false;
//}
//
//tempalte <typename Scalar, int Dim>
//void EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >::sortActiveDomainCorners()
//{
//  //TO DO
//}

//explicit instantiations
template class EnrichedMPMUniformGridGeneralizedVector<Vector<float,2> >;
template class EnrichedMPMUniformGridGeneralizedVector<Vector<float,3> >;
template class EnrichedMPMUniformGridGeneralizedVector<Vector<double,2> >;
template class EnrichedMPMUniformGridGeneralizedVector<Vector<double,3> >;

}  //end of namespace Physika
