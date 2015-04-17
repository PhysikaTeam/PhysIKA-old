/*
 * @file quad_cubic_mesh_fem_solid_force_model.h 
 * @Brief fem solid force model for quad and cubic mesh.
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

#ifndef PHYSIKA_DYNAMICS_FEM_FEM_SOLID_FORCE_MODEL_QUAD_CUBIC_MESH_FEM_SOLID_FORCE_MODEL_H_
#define PHYSIKA_DYNAMICS_FEM_FEM_SOLID_FORCE_MODEL_QUAD_CUBIC_MESH_FEM_SOLID_FORCE_MODEL_H_

#include <vector>
#include "Physika_Dynamics/FEM/FEM_Solid_Force_Model/fem_solid_force_model.h"

namespace Physika{

template <typename Scalar, int Dim> class VolumetricMesh;
template <typename Scalar, int Dim> class ConstitutiveModel;
template <typename Scalar, int Dim> class Vector;

template <typename Scalar, int Dim>
class QuadCubicMeshFEMSolidForceModel: public FEMSolidForceModel<Scalar,Dim>
{
public:
    QuadCubicMeshFEMSolidForceModel(const VolumetricMesh<Scalar,Dim> &simulation_mesh, const std::vector<ConstitutiveModel<Scalar,Dim>*> &constitutive_model);
    ~QuadCubicMeshFEMSolidForceModel();

    //given world space coordinates of mesh vertices, compute the internal forces on the entire mesh
    virtual void computeGlobalInternalForces(const std::vector<Vector<Scalar,Dim> > &current_vert_pos, std::vector<Vector<Scalar,Dim> > &force) const;
    //compute internal forces on vertices of a specific element
    virtual void computeElementInternalForces(unsigned int ele_idx, const std::vector<Vector<Scalar,Dim> > &current_vert_pos, std::vector<Vector<Scalar,Dim> > &force) const;
    //compute force differentials, for implicit time stepping
    virtual void computeGlobalInternalForceDifferentials(const std::vector<Vector<Scalar,Dim> > &current_vert_pos, 
                                                                                    const std::vector<Vector<Scalar,Dim> > &vert_pos_differentials,
                                                                                     std::vector<Vector<Scalar,Dim> > &force_differentials) const;
    virtual void computeElementInternalForceDifferentials(unsigned int ele_idx, const std::vector<Vector<Scalar,Dim> > &current_vert_pos,
                                                                                       const std::vector<Vector<Scalar,Dim> > &vert_pos_differentials,
                                                                                       std::vector<Vector<Scalar,Dim> > &force_differentials) const;
protected:
    QuadCubicMeshFEMSolidForceModel();
};
    
} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_FEM_FEM_SOLID_FORCE_MODEL_QUAD_CUBIC_MESH_FEM_SOLID_FORCE_MODEL_H_
