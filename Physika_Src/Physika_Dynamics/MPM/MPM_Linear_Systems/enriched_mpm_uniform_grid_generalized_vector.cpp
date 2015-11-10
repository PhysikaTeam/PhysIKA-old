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

#include "Physika_Dynamics/MPM/MPM_Linear_Systems/enriched_mpm_uniform_grid_generalized_vector.h"

namespace Physika{

// template <typename Scalar, int Dim>
// EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >::EnrichedMPMUniformGridGeneralizedVector
//                                                               (const Vector<unsigned int, Dim> &grid_size,
//                                                                unsigned int particle_num)
// {
//     //TO DO
// }
//
// template <typename Scalar, int Dim>
// EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >::EnrichedMPMUniformGridGeneralizedVector
//                                        (const Vector<unsigned int, Dim> &grid_size,
//                                         unsigned int particle_num,
//                                         const std::vector<Vector<unsigned, Dim> > &active_grid_nodes,
//                                         const std::vector<Scalar> &active_node_mass,
//                                         const std::vector<Vector<unsigned int, 2> >&active_domain_corners,
//                                         const std::vector<Scalar> &active_domain_corner_mass)
// {
//     //TO DO
// }



//explicit instantiations
template class EnrichedMPMUniformGridGeneralizedVector<Vector<float,2> >;
template class EnrichedMPMUniformGridGeneralizedVector<Vector<float,3> >;
template class EnrichedMPMUniformGridGeneralizedVector<Vector<double,2> >;
template class EnrichedMPMUniformGridGeneralizedVector<Vector<double,3> >;

}  //end of namespace Physika
