/*
 * @file fem_base.cpp 
 * @Brief Base class of FEM drivers, all FEM methods inherit from it.
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

#include <cstdlib>
#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_IO/Volumetric_Mesh_IO/volumetric_mesh_io.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh_internal.h"
#include "Physika_Geometry/Volumetric_Meshes/tri_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/quad_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/tet_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/cubic_mesh.h"
#include "Physika_Dynamics/FEM/fem_base.h"

namespace Physika{

template <typename Scalar, int Dim>
FEMBase<Scalar,Dim>::FEMBase()
    :DriverBase<Scalar>(),simulation_mesh_(NULL),gravity_(9.8)
{
}

template <typename Scalar, int Dim>
FEMBase<Scalar,Dim>::FEMBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :DriverBase<Scalar>(start_frame,end_frame,frame_rate,max_dt,write_to_file),simulation_mesh_(NULL),gravity_(9.8)
{
}

template <typename Scalar, int Dim>
FEMBase<Scalar,Dim>::FEMBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file,
                             const VolumetricMesh<Scalar,Dim> &mesh)
    :DriverBase<Scalar>(start_frame,end_frame,frame_rate,max_dt,write_to_file),simulation_mesh_(NULL),gravity_(9.8)
{
    setSimulationMesh(mesh);
}

template <typename Scalar, int Dim>
FEMBase<Scalar,Dim>::~FEMBase()
{
    if(simulation_mesh_)
        delete simulation_mesh_;
}

template <typename Scalar, int Dim>
Scalar FEMBase<Scalar,Dim>::gravity() const
{
    return gravity_;
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setGravity(Scalar gravity)
{
    gravity_ = gravity;
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::loadSimulationMesh(const std::string &file_name)
{
    if(simulation_mesh_)
        delete simulation_mesh_;
    simulation_mesh_ = VolumetricMeshIO<Scalar,Dim>::load(file_name);
    if(simulation_mesh_ == NULL)
    {
        std::cerr<<"Failed to load simulation mesh from "<<file_name<<"\n";
        std::exit(EXIT_FAILURE);
    }
    synchronizeDataWithSimulationMesh();
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setSimulationMesh(const VolumetricMesh<Scalar,Dim> &mesh)
{
    if(simulation_mesh_)
        delete simulation_mesh_;
    const VolumetricMesh<Scalar,Dim> *temp_ptr = &mesh;
    VolumetricMeshInternal::ElementType ele_type = mesh.elementType();
    switch(ele_type)
    {
    case VolumetricMeshInternal::TRI:
        simulation_mesh_ = dynamic_cast<VolumetricMesh<Scalar,Dim>*>(
            new TriMesh<Scalar>(*(dynamic_cast<const TriMesh<Scalar>*>(temp_ptr)))
            );
        break;
    case VolumetricMeshInternal::QUAD:
        simulation_mesh_ = dynamic_cast<VolumetricMesh<Scalar,Dim>*>(
            new QuadMesh<Scalar>(*(dynamic_cast<const QuadMesh<Scalar>*>(temp_ptr)))
            );
        break;
    case VolumetricMeshInternal::TET:
        simulation_mesh_ = dynamic_cast<VolumetricMesh<Scalar,Dim>*>(
            new TetMesh<Scalar>(*(dynamic_cast<const TetMesh<Scalar>*>(temp_ptr)))
            );
        break;
    case VolumetricMeshInternal::CUBIC:
        simulation_mesh_ = dynamic_cast<VolumetricMesh<Scalar,Dim>*>(
            new CubicMesh<Scalar>(*(dynamic_cast<const CubicMesh<Scalar>*>(temp_ptr)))
            );
        break;
    case VolumetricMeshInternal::NON_UNIFORM:
        PHYSIKA_ERROR("Non-uniform element type not implemented yet.");
        break;
    default:
        PHYSIKA_ERROR("Unknown element type.");
        break;
    }
    synchronizeDataWithSimulationMesh();
}

template <typename Scalar, int Dim>
const VolumetricMesh<Scalar,Dim>* FEMBase<Scalar,Dim>::simulationMesh() const
{
    return simulation_mesh_;
}

template <typename Scalar, int Dim>
VolumetricMesh<Scalar,Dim>* FEMBase<Scalar,Dim>::simulationMesh()
{
    return simulation_mesh_;
}

template <typename Scalar, int Dim>
unsigned int FEMBase<Scalar,Dim>::numSimVertices() const
{
    if(simulation_mesh_==NULL)
    {
        std::cerr<<"Simulation mesh not set.\n";
        std::exit(EXIT_FAILURE);
    }
    return simulation_mesh_->vertNum();
}

template <typename Scalar, int Dim>
unsigned int FEMBase<Scalar,Dim>::numSimElements() const
{
    if(simulation_mesh_==NULL)
    {
        std::cerr<<"Simulation mesh not set.\n";
        std::exit(EXIT_FAILURE);
    }
    return simulation_mesh_->eleNum();
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> FEMBase<Scalar,Dim>::vertexDisplacement(unsigned int vert_idx) const
{
    if(vert_idx >= vertex_displacements_.size())
    {
        std::cerr<<"Vertex index out of range.\n";
        std::exit(EXIT_FAILURE);
    }
    return vertex_displacements_[vert_idx];
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setVertexDisplacement(unsigned int vert_idx, const Vector<Scalar,Dim> &u)
{
    if(vert_idx >= vertex_displacements_.size())
    {
        std::cerr<<"Vertex index out of range.\n";
        std::exit(EXIT_FAILURE);
    }
    vertex_displacements_[vert_idx] = u;
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::resetVertexDisplacement()
{
    for(unsigned int i = 0; i < vertex_displacements_.size(); ++i)
        vertex_displacements_[i] = Vector<Scalar,Dim>(0);
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> FEMBase<Scalar,Dim>::vertexRestPosition(unsigned int vert_idx) const
{
    if(simulation_mesh_==NULL)
    {
        std::cerr<<"Simulation mesh not set.\n";
        std::exit(EXIT_FAILURE);
    }
    if(vert_idx >= vertex_displacements_.size())
    {
        std::cerr<<"Vertex index out of range.\n";
        std::exit(EXIT_FAILURE);
    }
    return simulation_mesh_->vertPos(vert_idx);
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> FEMBase<Scalar,Dim>::vertexCurrentPosition(unsigned int vert_idx) const
{
    if(simulation_mesh_==NULL)
    {
        std::cerr<<"Simulation mesh not set.\n";
        std::exit(EXIT_FAILURE);
    }
    if(vert_idx >= vertex_displacements_.size())
    {
        std::cerr<<"Vertex index out of range.\n";
        std::exit(EXIT_FAILURE);
    }
    return simulation_mesh_->vertPos(vert_idx) + vertex_displacements_[vert_idx];  
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> FEMBase<Scalar,Dim>::vertexVelocity(unsigned int vert_idx) const
{
    if(vert_idx >= vertex_displacements_.size())
    {
        std::cerr<<"Vertex index out of range.\n";
        std::exit(EXIT_FAILURE);
    }
    return vertex_velocities_[vert_idx];
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setVertexVelocity(unsigned int vert_idx, const Vector<Scalar,Dim> &v)
{
    if(simulation_mesh_==NULL)
    {
        std::cerr<<"Simulation mesh not set.\n";
        std::exit(EXIT_FAILURE);
    }
    if(vert_idx >= vertex_displacements_.size())
    {
        std::cerr<<"Vertex index out of range.\n";
        std::exit(EXIT_FAILURE);
    }
    vertex_velocities_[vert_idx] = v;
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::resetVertexVelocity()
{
    for(unsigned int i = 0; i < vertex_velocities_.size(); ++i)
        vertex_velocities_[i] = Vector<Scalar,Dim>(0);
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::synchronizeDataWithSimulationMesh()
{
    vertex_displacements_.resize(simulation_mesh_->vertNum());
    vertex_velocities_.resize(simulation_mesh_->vertNum());
    computeReferenceShapeMatrixInverse();
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> FEMBase<Scalar,Dim>::deformationGradient(unsigned int ele_idx) const
{
//TO DO
    unsigned int ele_num = simulation_mesh_->eleNum();
    PHYSIKA_ASSERT(ele_idx<ele_num);
    VolumetricMeshInternal::ElementType ele_type = simulation_mesh_->elementType();
    switch(ele_type)
    {
    case VolumetricMeshInternal::TRI:
        break;
    case VolumetricMeshInternal::QUAD:
        break;
    case VolumetricMeshInternal::TET:
        break;
    case VolumetricMeshInternal::CUBIC:
        break;
    case VolumetricMeshInternal::NON_UNIFORM:
        PHYSIKA_ERROR("Non-uniform element type not implemented yet.");
        break;
    default:
        PHYSIKA_ERROR("Unknown element type.");
        break;
    }
    return SquareMatrix<Scalar,Dim>();
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::computeReferenceShapeMatrixInverse()
{
    reference_shape_matrix_inv_.clear();
    VolumetricMeshInternal::ElementType ele_type = simulation_mesh_->elementType();
    unsigned int ele_num = simulation_mesh_->eleNum();
    switch(ele_type)
    {
    case VolumetricMeshInternal::TRI:
    {
        std::vector<Vector<Scalar,2> > ele_vert_pos;
        for(unsigned int i = 0; i < ele_num; ++i)
        {
            ele_vert_pos.clear();
            TriMesh<Scalar> *tri_mesh = dynamic_cast<TriMesh<Scalar>*>(simulation_mesh_);
            PHYSIKA_ASSERT(tri_mesh);
            tri_mesh->eleVertPos(i,ele_vert_pos);
            PHYSIKA_ASSERT(ele_vert_pos.size()==3);
            Vector<Scalar,2> v1_minus_v3 = ele_vert_pos[0] - ele_vert_pos[2];
            Vector<Scalar,2> v2_minus_v3 = ele_vert_pos[1] - ele_vert_pos[2];
            SquareMatrix<Scalar,2> reference_shape_matrix(v1_minus_v3,v2_minus_v3);
            reference_shape_matrix = reference_shape_matrix.transpose();
            SquareMatrix<Scalar,2> inv = reference_shape_matrix.inverse();
            SquareMatrix<Scalar,Dim> *mat_ptr = dynamic_cast<SquareMatrix<Scalar,Dim>*>(&inv);
            PHYSIKA_ASSERT(mat_ptr);
            reference_shape_matrix_inv_.push_back(*mat_ptr);
        }
        break;
    }
    case VolumetricMeshInternal::QUAD:
    {
        break;
    }
    case VolumetricMeshInternal::TET:
    {
        std::vector<Vector<Scalar,3> > ele_vert_pos;
        for(unsigned int i = 0; i < ele_num; ++i)
        {
            ele_vert_pos.clear();
            TetMesh<Scalar> *tet_mesh = dynamic_cast<TetMesh<Scalar>*>(simulation_mesh_);
            PHYSIKA_ASSERT(tet_mesh);
            tet_mesh->eleVertPos(i,ele_vert_pos);
            PHYSIKA_ASSERT(ele_vert_pos.size()==4);
            Vector<Scalar,3> v1_minus_v4 = ele_vert_pos[0] - ele_vert_pos[3];
            Vector<Scalar,3> v2_minus_v4 = ele_vert_pos[1] - ele_vert_pos[3];
            Vector<Scalar,3> v3_minus_v4 = ele_vert_pos[2] - ele_vert_pos[3];
            SquareMatrix<Scalar,3> reference_shape_matrix(v1_minus_v4,v2_minus_v4,v3_minus_v4);
            SquareMatrix<Scalar,3> inv = reference_shape_matrix.inverse();
            SquareMatrix<Scalar,Dim> *mat_ptr = dynamic_cast<SquareMatrix<Scalar,Dim>*>(&inv);
            PHYSIKA_ASSERT(mat_ptr);
            reference_shape_matrix_inv_.push_back(*mat_ptr);
        }
        break;
    }
    case VolumetricMeshInternal::CUBIC:
    {
        break;
    }
    case VolumetricMeshInternal::NON_UNIFORM:
        PHYSIKA_ERROR("Non-uniform element type not implemented yet.");
        break;
    default:
        PHYSIKA_ERROR("Unknown element type.");
        break;
    }
//TO DO
}

//explicit instantiations
template class FEMBase<float,2>;
template class FEMBase<float,3>;
template class FEMBase<double,2>;
template class FEMBase<double,3>;

}  //end of namespace Physika
