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
}

//use explicit specialization of member functions for setSimulationMesh()
template <> 
void FEMBase<float,2>::setSimulationMesh(const VolumetricMesh<float,2> &mesh)
{
    if(simulation_mesh_)
        delete simulation_mesh_;
    const VolumetricMesh<float,2> *temp_ptr = &mesh;
    VolumetricMeshInternal::ElementType ele_type = mesh.elementType();
    switch(ele_type)
    {
    case VolumetricMeshInternal::TRI:
        simulation_mesh_ = new TriMesh<float>(*(dynamic_cast<const TriMesh<float>*>(temp_ptr)));
        break;
    case VolumetricMeshInternal::QUAD:
        simulation_mesh_ = new QuadMesh<float>(*(dynamic_cast<const QuadMesh<float>*>(temp_ptr)));
        break;
    case VolumetricMeshInternal::TET:
    case VolumetricMeshInternal::CUBIC:
        PHYSIKA_ERROR("ERROR LOGIC.");
        break;
    case VolumetricMeshInternal::NON_UNIFORM:
        PHYSIKA_ERROR("Non-uniform element type not implemented yet.");
        break;
    default:
        PHYSIKA_ERROR("Unknown element type.");
        break;
    }
}
template <>
void FEMBase<double,2>::setSimulationMesh(const VolumetricMesh<double,2> &mesh)
{
    if(simulation_mesh_)
        delete simulation_mesh_;
    const VolumetricMesh<double,2> *temp_ptr = &mesh;
    VolumetricMeshInternal::ElementType ele_type = mesh.elementType();
    switch(ele_type)
    {
    case VolumetricMeshInternal::TRI:
        simulation_mesh_ = new TriMesh<double>(*(dynamic_cast<const TriMesh<double>*>(temp_ptr)));
        break;
    case VolumetricMeshInternal::QUAD:
        simulation_mesh_ = new QuadMesh<double>(*(dynamic_cast<const QuadMesh<double>*>(temp_ptr)));
        break;
    case VolumetricMeshInternal::TET:
    case VolumetricMeshInternal::CUBIC:
        PHYSIKA_ERROR("ERROR LOGIC.");
        break;
    case VolumetricMeshInternal::NON_UNIFORM:
        PHYSIKA_ERROR("Non-uniform element type not implemented yet.");
        break;
    default:
        PHYSIKA_ERROR("Unknown element type.");
        break;
    }
}
template <>
void FEMBase<float,3>::setSimulationMesh(const VolumetricMesh<float,3> &mesh)
{
    if(simulation_mesh_)
        delete simulation_mesh_;
    const VolumetricMesh<float,3> *temp_ptr = &mesh;
    VolumetricMeshInternal::ElementType ele_type = mesh.elementType();
    switch(ele_type)
    {
    case VolumetricMeshInternal::TRI:
    case VolumetricMeshInternal::QUAD:
        PHYSIKA_ERROR("ERROR LOGIC.");
        break;
    case VolumetricMeshInternal::TET:
        simulation_mesh_ = new TetMesh<float>(*(dynamic_cast<const TetMesh<float>*>(temp_ptr)));
        break;
    case VolumetricMeshInternal::CUBIC:
        simulation_mesh_ = new CubicMesh<float>(*(dynamic_cast<const CubicMesh<float>*>(temp_ptr)));
        break;
    case VolumetricMeshInternal::NON_UNIFORM:
        PHYSIKA_ERROR("Non-uniform element type not implemented yet.");
        break;
    default:
        PHYSIKA_ERROR("Unknown element type.");
        break;
    }
}
template <>
void FEMBase<double,3>::setSimulationMesh(const VolumetricMesh<double,3> &mesh)
{
    if(simulation_mesh_)
        delete simulation_mesh_;
    const VolumetricMesh<double,3> *temp_ptr = &mesh;
    VolumetricMeshInternal::ElementType ele_type = mesh.elementType();
    switch(ele_type)
    {
    case VolumetricMeshInternal::TRI:
    case VolumetricMeshInternal::QUAD:
        PHYSIKA_ERROR("ERROR LOGIC.");
        break;
    case VolumetricMeshInternal::TET:
        simulation_mesh_ = new TetMesh<double>(*(dynamic_cast<const TetMesh<double>*>(temp_ptr)));
        break;
    case VolumetricMeshInternal::CUBIC:
        simulation_mesh_ = new CubicMesh<double>(*(dynamic_cast<const CubicMesh<double>*>(temp_ptr)));
        break;
    case VolumetricMeshInternal::NON_UNIFORM:
        PHYSIKA_ERROR("Non-uniform element type not implemented yet.");
        break;
    default:
        PHYSIKA_ERROR("Unknown element type.");
        break;
    }
}

//explicit instantiations
template class FEMBase<float,2>;
template class FEMBase<float,3>;
template class FEMBase<double,2>;
template class FEMBase<double,3>;

}  //end of namespace Physika
