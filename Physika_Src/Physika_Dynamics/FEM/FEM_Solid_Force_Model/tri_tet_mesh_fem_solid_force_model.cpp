/*
 * @file tri_tet_mesh_fem_solid_force_model.cpp
 * @Brief fem solid force model for constant strain triangle and tetrahedron mesh.
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

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh_internal.h"
#include "Physika_Geometry/Volumetric_Meshes/tri_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/tet_mesh.h"
#include "Physika_Dynamics/Constitutive_Models/constitutive_model.h"
#include "Physika_Dynamics/FEM/FEM_Solid_Force_Model/tri_tet_mesh_fem_solid_force_model.h"

namespace Physika{

template <typename Scalar, int Dim>
TriTetMeshFEMSolidForceModel<Scalar,Dim>::TriTetMeshFEMSolidForceModel(const VolumetricMesh<Scalar,Dim> &simulation_mesh,
                                                                       const std::vector<ConstitutiveModel<Scalar,Dim>*> &constitutive_model)
    :FEMSolidForceModel<Scalar,Dim>(simulation_mesh,constitutive_model)
{
    updatePrecomputedData();
}

template <typename Scalar, int Dim>
TriTetMeshFEMSolidForceModel<Scalar,Dim>::~TriTetMeshFEMSolidForceModel()
{
}

template <typename Scalar, int Dim>
void TriTetMeshFEMSolidForceModel<Scalar,Dim>::updatePrecomputedData()
{
    //first check if the volumetric mesh is TriMesh or TetMesh
    const VolumetricMesh<Scalar,Dim> &sim_mesh = (this->simulation_mesh_);
    VolumetricMeshInternal::ElementType ele_type = sim_mesh.elementType();
    if((ele_type != VolumetricMeshInternal::TRI) && (ele_type != VolumetricMeshInternal::TET))
        throw PhysikaException("Simulation mesh element type and FEM force model mismatch!");
    computeReferenceElementVolume();
    computeReferenceShapeMatrixInverse();
}

template <typename Scalar, int Dim>
void TriTetMeshFEMSolidForceModel<Scalar,Dim>::computeGlobalInternalForces(const std::vector<Vector<Scalar,Dim> > &current_vert_pos, std::vector<Vector<Scalar,Dim> > &force) const
{
    const VolumetricMesh<Scalar,Dim> &sim_mesh = (this->simulation_mesh_);
    unsigned int vert_num = sim_mesh.vertNum();
    if(current_vert_pos.size() != vert_num)
        throw PhysikaException("Size of provided vertex position vector doesn't match mesh vertex number!");
    force.resize(vert_num,Vector<Scalar,Dim>(0));
    unsigned int ele_num = sim_mesh.eleNum();
    std::vector<Vector<Scalar,Dim> > ele_force;
    std::vector<Vector<Scalar,Dim> > ele_vert_pos;
    for(unsigned int ele_idx = 0; ele_idx < ele_num; ++ele_idx)
    {
        unsigned int ele_vert_num = sim_mesh.eleVertNum(ele_idx);
        ele_vert_pos.resize(ele_vert_num);
        for (unsigned int local_vert_idx = 0; local_vert_idx < ele_vert_num; ++local_vert_idx)
        {
            unsigned int vert_idx = sim_mesh.eleVertIndex(ele_idx, local_vert_idx);
            ele_vert_pos[local_vert_idx] = current_vert_pos[vert_idx];
        }
        computeElementInternalForces(ele_idx,ele_vert_pos,ele_force);
        for(unsigned int local_vert_idx = 0; local_vert_idx < ele_vert_num; ++local_vert_idx)
        {
            unsigned int vert_idx = sim_mesh.eleVertIndex(ele_idx,local_vert_idx);
            force[vert_idx] += ele_force[local_vert_idx];
        }
    }
}

template <typename Scalar, int Dim>
void TriTetMeshFEMSolidForceModel<Scalar,Dim>::computeElementInternalForces(unsigned int ele_idx,
                                        const std::vector<Vector<Scalar,Dim> > &current_vert_pos,
                                        std::vector<Vector<Scalar,Dim> > &force) const
{
    const VolumetricMesh<Scalar,Dim> &sim_mesh = (this->simulation_mesh_);
    unsigned int ele_num = sim_mesh.eleNum();
    if(ele_idx >= ele_num)
        throw PhysikaException("Simulation mesh element index out of range!");
    unsigned int ele_vert_num = sim_mesh.eleVertNum(ele_idx);
    if(current_vert_pos.size() != ele_vert_num)
        throw PhysikaException("Size of provided vertex position vector doesn't match element vertex number!");
    VolumetricMeshInternal::ElementType ele_type = sim_mesh.elementType();
    SquareMatrix<Scalar,Dim> shape_matrix;
    switch(ele_type)
    {
    case VolumetricMeshInternal::TRI:
    {
        std::vector<Vector<Scalar,2> > current_vert_pos_trait(current_vert_pos.size());
        for(unsigned int vert_idx = 0; vert_idx < current_vert_pos.size(); ++vert_idx)
        {
            current_vert_pos_trait[vert_idx][0] = current_vert_pos[vert_idx][0];
            current_vert_pos_trait[vert_idx][1] = current_vert_pos[vert_idx][1];
        }
        Vector<Scalar,2> v1_minus_v3 = current_vert_pos_trait[0] - current_vert_pos_trait[2];
        Vector<Scalar,2> v2_minus_v3 = current_vert_pos_trait[1] - current_vert_pos_trait[2];
        SquareMatrix<Scalar,2> current_shape_matrix(v1_minus_v3,v2_minus_v3);
        current_shape_matrix = current_shape_matrix.transpose();
        for(unsigned int i = 0; i < 2; ++i)
            for(unsigned int j =0; j < 2; ++j)
                shape_matrix(i,j) = current_shape_matrix(i,j);
        break;
    }
    case VolumetricMeshInternal::TET:
    {
        std::vector<Vector<Scalar,3> > current_vert_pos_trait(current_vert_pos.size());
        for(unsigned int vert_idx = 0; vert_idx < current_vert_pos.size(); ++vert_idx)
        {
            current_vert_pos_trait[vert_idx][0] = current_vert_pos[vert_idx][0];
            current_vert_pos_trait[vert_idx][1] = current_vert_pos[vert_idx][1];
            current_vert_pos_trait[vert_idx][2] = current_vert_pos[vert_idx][2];
        }
        Vector<Scalar,3> v1_minus_v4 = current_vert_pos_trait[0] - current_vert_pos_trait[3];
        Vector<Scalar,3> v2_minus_v4 = current_vert_pos_trait[1] - current_vert_pos_trait[3];
        Vector<Scalar,3> v3_minus_v4 = current_vert_pos_trait[2] - current_vert_pos_trait[3];
        SquareMatrix<Scalar,3> current_shape_matrix(v1_minus_v4,v2_minus_v4,v3_minus_v4);
        current_shape_matrix = current_shape_matrix.transpose();
        for(unsigned int i = 0; i < 3; ++i)
            for(unsigned int j = 0; j < 3; ++j)
                shape_matrix(i,j) = current_shape_matrix(i,j);
        break;
    }
    case VolumetricMeshInternal::QUAD:
    case VolumetricMeshInternal::CUBIC:
    case VolumetricMeshInternal::NON_UNIFORM:
        throw PhysikaException("Simulation mesh element type and FEM force model mismatch!");
    }
    SquareMatrix<Scalar,Dim> deform_grad = shape_matrix*reference_shape_matrix_inv_[ele_idx];
    const ConstitutiveModel<Scalar,Dim> &ele_material = this->elementMaterial(ele_idx);
    SquareMatrix<Scalar,Dim> first_Piola_Kirchhoff_stress = ele_material.firstPiolaKirchhoffStress(deform_grad);
    SquareMatrix<Scalar,Dim> force_as_col = (-1.0)*reference_element_volume_[ele_idx]*first_Piola_Kirchhoff_stress*(reference_shape_matrix_inv_[ele_idx].transpose());
    force.resize(ele_vert_num,Vector<Scalar,Dim>(0));
    for(unsigned int i = 0; i < ele_vert_num - 1; ++i)
    {
        force[i] = force_as_col.colVector(i);
        force[ele_vert_num-1] -= force[i];
    }
}

template <typename Scalar, int Dim>
void TriTetMeshFEMSolidForceModel<Scalar,Dim>::computeGlobalInternalForceDifferentials(
                                               const std::vector<Vector<Scalar,Dim> > &current_vert_pos,
                                               const std::vector<Vector<Scalar,Dim> > &vert_pos_differentials,
                                               std::vector<Vector<Scalar,Dim> > &force_differentials) const
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar, int Dim>
void TriTetMeshFEMSolidForceModel<Scalar,Dim>::computeElementInternalForceDifferentials(
                                                unsigned int ele_idx,
                                                const std::vector<Vector<Scalar,Dim> > &current_vert_pos,
                                                const std::vector<Vector<Scalar,Dim> > &vert_pos_differentials,
                                                std::vector<Vector<Scalar,Dim> > &force_differentials) const
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar, int Dim>
void TriTetMeshFEMSolidForceModel<Scalar,Dim>::computeReferenceElementVolume()
{
    const VolumetricMesh<Scalar,Dim> &sim_mesh = (this->simulation_mesh_);
    unsigned int ele_num = sim_mesh.eleNum();
    reference_element_volume_.resize(ele_num);
    for(unsigned int ele_idx = 0; ele_idx < ele_num; ++ele_idx)
        reference_element_volume_[ele_idx] = sim_mesh.eleVolume(ele_idx);
}

template <typename Scalar, int Dim>
void TriTetMeshFEMSolidForceModel<Scalar,Dim>::computeReferenceShapeMatrixInverse()
{
    const VolumetricMesh<Scalar,Dim> &sim_mesh = (this->simulation_mesh_);
    unsigned int ele_num = sim_mesh.eleNum();
    reference_shape_matrix_inv_.resize(ele_num);
    VolumetricMeshInternal::ElementType ele_type = sim_mesh.elementType();
    switch(ele_type)
    {
    case VolumetricMeshInternal::TRI:
    {
        std::vector<Vector<Scalar,2> > ele_vert_pos;
        const TriMesh<Scalar> *tri_mesh = dynamic_cast<const TriMesh<Scalar>*>(&sim_mesh);
        for(unsigned int ele_idx = 0; ele_idx < ele_num; ++ele_idx)
        {
            ele_vert_pos.clear();
            tri_mesh->eleVertPos(ele_idx,ele_vert_pos);
            Vector<Scalar,2> v1_minus_v3 = ele_vert_pos[0] - ele_vert_pos[2];
            Vector<Scalar,2> v2_minus_v3 = ele_vert_pos[1] - ele_vert_pos[2];
            SquareMatrix<Scalar,2> reference_shape_matrix(v1_minus_v3,v2_minus_v3);
            reference_shape_matrix = reference_shape_matrix.transpose();
            SquareMatrix<Scalar,2> inv = reference_shape_matrix.inverse();
            for(unsigned int i = 0; i < 2; ++i)
                for(unsigned int j = 0; j < 2; ++j)
                    reference_shape_matrix_inv_[ele_idx](i,j) = inv(i,j);
        }
        break;
    }
    case VolumetricMeshInternal::TET:
    {
        std::vector<Vector<Scalar,3> > ele_vert_pos;
        const TetMesh<Scalar> *tet_mesh = dynamic_cast<const TetMesh<Scalar>*>(&sim_mesh);
        for(unsigned int ele_idx = 0; ele_idx < ele_num; ++ele_idx)
        {
            ele_vert_pos.clear();
            tet_mesh->eleVertPos(ele_idx,ele_vert_pos);
            Vector<Scalar,3> v1_minus_v4 = ele_vert_pos[0] - ele_vert_pos[3];
            Vector<Scalar,3> v2_minus_v4 = ele_vert_pos[1] - ele_vert_pos[3];
            Vector<Scalar,3> v3_minus_v4 = ele_vert_pos[2] - ele_vert_pos[3];
            SquareMatrix<Scalar,3> reference_shape_matrix(v1_minus_v4,v2_minus_v4,v3_minus_v4);
            reference_shape_matrix = reference_shape_matrix.transpose();
            SquareMatrix<Scalar,3> inv = reference_shape_matrix.inverse();
            for(unsigned int i = 0; i < 3; ++i)
                for(unsigned int j = 0; j < 3; ++j)
                    reference_shape_matrix_inv_[ele_idx](i,j) = inv(i,j);
        }
        break;
    }
    case VolumetricMeshInternal::QUAD:
    case VolumetricMeshInternal::CUBIC:
    case VolumetricMeshInternal::NON_UNIFORM:
        throw PhysikaException("Simulation mesh element type and FEM force model mismatch!");
    }
}

//explicit instantiations
template class TriTetMeshFEMSolidForceModel<float,2>;
template class TriTetMeshFEMSolidForceModel<float,3>;
template class TriTetMeshFEMSolidForceModel<double,2>;
template class TriTetMeshFEMSolidForceModel<double,3>;

}  //end of namespace Physika
