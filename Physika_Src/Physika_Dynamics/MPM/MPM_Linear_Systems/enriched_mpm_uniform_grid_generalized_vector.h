/*
* @file enriched_mpm_uniform_grid_generalized_vector.h
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

#ifndef PHYSIKA_DYNAMICS_MPM_MPM_LINEAR_SYSTEMS_ENRICHED_MPM_UNIFORM_GRID_GENERALIZED_VECTOR_H_
#define PHYSIKA_DYNAMICS_MPM_MPM_LINEAR_SYSTEMS_ENRICHED_MPM_UNIFORM_GRID_GENERALIZED_VECTOR_H_

#include "Physika_Dynamics/MPM/MPM_Linear_Systems/mpm_uniform_grid_generalized_vector.h"

namespace Physika{

/*
 * EnrichedMPMUniformGridGeneralizedVector: the generalized vector for mpm drivers enriched with
 * particle domain corners
 * the element is Vector<Scalar,Dim> where Dim = 2/3
 */

//default template, constructor made private to prohibit instance
template <typename Scalar>
class EnrichedMPMUniformGridGeneralizedVector
{
private:
    EnrichedMPMUniformGridGeneralizedVector();
};

////partial specialization for Vector<Scalar,Dim>
//template <typename Scalar, int Dim>
//class EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >: public MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >
//{
//public:
//    //all grid nodes active, node mass set to 1, no active enriched domain corners
//    explicit EnrichedMPMUniformGridGeneralizedVector(const Vector<unsigned int, Dim> &grid_size);
//    EnrichedMPMUniformGridGeneralizedVector(const Vector<unsigned int, Dim> &grid_size,
//                                            const std::vector<Vector<unsigned, Dim> > &active_grid_nodes,
//                                            const std::vector<Scalar> &active_node_mass,
//                                            const std::vector<Vector<unsigned int, 2> >&active_domain_corners,
//                                            const std::vector<Scalar> &active_domain_corner_mass);
//    EnrichedMPMUniformGridGeneralizedVector(const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &vector);
//    ~EnrichedMPMUniformGridGeneralizedVector();
//    EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >& operator= (const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &vector);
//    virtual EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >* clone() const;
//    virtual Scalar norm() const;
//    virtual Scalar normSquared() const;
//    virtual Scalar dot(const GeneralizedVector<Scalar> &vector) const;
//    const Vector<Scalar, Dim>& operator[](const Vector<unsigned int, Dim> &idx) const;
//    Vector<Scalar, Dim>& operator[](const Vector<unsigned int, Dim> &idx);
//    const Vector<Scalar, Dim>& operator[](unsigned int particle_idx, unsigned int corner_idx) const;
//    Vector<Scalar, Dim>& operator[](unsigned int particle_idx, unsigned int corner_idx);
//protected:
//    std::vector<Vector<unsigned int, 2> > active_particle_domain_corners_; //each pair represents [particle_idx, corner_idx]
//    std::vector<Scalar> active_domain_corner_mass_;
//};

}

#endif //PHYSIKA_DYNAMICS_MPM_MPM_LINEAR_SYSTEMS_ENRICHED_MPM_UNIFORM_GRID_GENERALIZED_VECTOR_H_