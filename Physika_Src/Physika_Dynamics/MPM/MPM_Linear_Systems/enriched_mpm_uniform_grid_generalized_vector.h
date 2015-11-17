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

#include <vector>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Numerics/Linear_System_Solvers/generalized_vector.h"
#include "Physika_Dynamics/Utilities/Grid_Generalized_Vectors/uniform_grid_generalized_vector_TV.h"

namespace Physika{

template <typename Scalar, int Dim> class VolumetricMesh;

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

//partial specialization for Vector<Scalar,Dim>
template <typename Scalar, int Dim>
class EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >: public GeneralizedVector<Scalar>
{
public:
    //all grid nodes active, no particles marked as enriched
    EnrichedMPMUniformGridGeneralizedVector(const Vector<unsigned int, Dim> &grid_size,
                                            const std::vector<VolumetricMesh<Scalar,Dim>*> &particle_domain_topology);
    EnrichedMPMUniformGridGeneralizedVector(const Vector<unsigned int, Dim> &grid_size,
                                            const std::vector<Vector<unsigned, Dim> > &active_grid_nodes,
                                            const std::vector<std::vector<unsigned int> > &enriched_particles,
                                            const std::vector<VolumetricMesh<Scalar,Dim>*> &particle_domain_topology);
    EnrichedMPMUniformGridGeneralizedVector(const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &vector);
    ~EnrichedMPMUniformGridGeneralizedVector();
    EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >& operator= (const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &vector);
    virtual EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >* clone() const;
    virtual unsigned int size() const;
    virtual EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >& operator+= (const GeneralizedVector<Scalar> &vector);
    virtual EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >& operator-= (const GeneralizedVector<Scalar> &vector);
    virtual EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >& operator*= (Scalar);
    virtual EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >& operator/= (Scalar);
    //accessors: data at grid nodes and domain corners
    const Vector<Scalar, Dim>& operator[](const Vector<unsigned int, Dim> &idx) const;
    Vector<Scalar, Dim>& operator[](const Vector<unsigned int, Dim> &idx);
    const Vector<Scalar, Dim>& operator()(unsigned int object_idx, unsigned int particle_idx, unsigned int corner_idx) const;
    Vector<Scalar, Dim>& operator()(unsigned int object_idx, unsigned int particle_idx, unsigned int corner_idx);
    void setValue(const Vector<Scalar,Dim> &value);
    void setActivePattern(const std::vector<Vector<unsigned int, Dim> > &active_grid_nodes,
                          const std::vector<std::vector<unsigned int> > &enriched_particles);
protected:
    EnrichedMPMUniformGridGeneralizedVector();
    virtual void copy(const GeneralizedVector<Scalar> &vector);
    bool checkActivePattern(const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >& vector) const; //check if active pattern matches
    void setEnrichedDomainCorners(const std::vector<)
protected:
    UniformGridGeneralizedVector<Vector<Scalar,Dim>,Dim> grid_data_;
    std::vector<std::vector<Vector<Scalar, Dim> > > domain_corner_data_;  //data at particle domain corners (mesh node of volumetric mesh)
    std::vector<std::vector<unsigned int> > enriched_domain_corners_;  //index of enriched domain corners (node index of volumetric mesh)
    std::vector<VolumetricMesh<Scalar, Dim>*> particle_domain_topology_;  //volumetric mesh representing topology of particle domains
};

}

#endif //PHYSIKA_DYNAMICS_MPM_MPM_LINEAR_SYSTEMS_ENRICHED_MPM_UNIFORM_GRID_GENERALIZED_VECTOR_H_
