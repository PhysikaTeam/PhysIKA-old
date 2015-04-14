/*
 * @file tri_tet_mesh_fem_solid_force_model.h 
 * @Brief fem solid force model for constant strain triangle and tetrahedron mesh.
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

#ifndef PHYSIKA_DYNAMICS_FEM_FEM_SOLID_FORCE_MODEL_TRI_TET_MESH_FEM_SOLID_FORCE_MODEL_H_
#define PHYSIKA_DYNAMICS_FEM_FEM_SOLID_FORCE_MODEL_TRI_TET_MESH_FEM_SOLID_FORCE_MODEL_H_

#include <vector>
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Dynamics/FEM/FEM_Solid_Force_Model/fem_solid_force_model.h"

namespace Physika{

template <typename Scalar, int Dim> class Vector;

template <typename Scalar, int Dim>
class TriTetMeshFEMSolidForceModel: public FEMSolidForceModel<Scalar,Dim>
{
public:
    TriTetMeshFEMSolidForceModel();
    explicit TriTetMeshFEMSolidForceModel(const FEMSolid<Scalar,Dim> *fem_solid_driver);
    ~TriTetMeshFEMSolidForceModel();
    virtual void setDriver(const FEMSolid<Scalar,Dim> *fem_solid_driver); //precomputed data is updated correspondently
    void updatePrecomputedData(); //whenever the volumetric mesh is modified, this method needs to be called to update the precomputed data
    //given world space coordinates of mesh vertices, compute the internal forces on the entire mesh
    virtual void computeGlobalInternalForces(const std::vector<Vector<Scalar,Dim> > &current_vert_pos, std::vector<Vector<Scalar,Dim> > &force) const;
    //compute internal forces on vertices of a specific element
    virtual void computeElementInternalForces(unsigned int ele_idx, const std::vector<Vector<Scalar,Dim> > &current_vert_pos, std::vector<Vector<Scalar,Dim> > &force) const;
    //compute force differentials, for implicit time stepping
    virtual void computeGlobalInternalForceDifferentials(const std::vector<Vector<Scalar,Dim> > &current_vert_pos, const std::vector<Vector<Scalar,Dim> > &vert_pos_differentials,
                                                                          std::vector<Vector<Scalar,Dim> > &force_differentials) const;
    virtual void computeElementInternalForceDifferentials(unsigned int ele_idx, const std::vector<Vector<Scalar,Dim> > &current_vert_pos, const std::vector<Vector<Scalar,Dim> > &vert_pos_differentials,
                                                                             std::vector<Vector<Scalar,Dim> > &force_differentials) const;
protected:
    void computeReferenceElementVolume();
    void computeReferenceShapeMatrixInverse();
protected:
    std::vector<Scalar> reference_element_volume_;
    std::vector<SquareMatrix<Scalar,Dim> > reference_shape_matrix_inv_;//store precomputed data (inverse of Dm) for deformation gradient computation: F = Ds*inv(Dm)
};
    
} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_FEM_FEM_SOLID_FORCE_MODEL_TRI_TET_MESH_FEM_SOLID_FORCE_MODEL_H_
