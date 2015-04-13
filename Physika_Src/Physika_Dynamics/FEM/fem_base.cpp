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

#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/physika_exception.h"
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
        std::string err("Failed to load simulation mesh from ");
        err += file_name;
        throw PhysikaException(err);
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
        throw PhysikaException("FEM with non-uniform element type not implemented yet.");
        break;
    default:
        PHYSIKA_ERROR("Unknown element type.");
        break;
    }
    synchronizeDataWithSimulationMesh();
}

template <typename Scalar, int Dim>
const VolumetricMesh<Scalar,Dim>& FEMBase<Scalar,Dim>::simulationMesh() const
{
    if(simulation_mesh_==NULL)
        throw PhysikaException("Simulation mesh not set.");
    return *simulation_mesh_;
}

template <typename Scalar, int Dim>
VolumetricMesh<Scalar,Dim>& FEMBase<Scalar,Dim>::simulationMesh()
{
    if(simulation_mesh_==NULL)
        throw PhysikaException("Simulation mesh not set.");
    return *simulation_mesh_;
}

template <typename Scalar, int Dim>
unsigned int FEMBase<Scalar,Dim>::numSimVertices() const
{
    if(simulation_mesh_==NULL)
        throw PhysikaException("Simulation mesh not set.");
    return simulation_mesh_->vertNum();
}

template <typename Scalar, int Dim>
unsigned int FEMBase<Scalar,Dim>::numSimElements() const
{
    if(simulation_mesh_==NULL)
        throw PhysikaException("Simulation mesh not set.");
    return simulation_mesh_->eleNum();
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> FEMBase<Scalar,Dim>::vertexDisplacement(unsigned int vert_idx) const
{
    if(vert_idx >= vertex_displacements_.size())
        throw PhysikaException("Vertex index out of range.");
    return vertex_displacements_[vert_idx];
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setVertexDisplacement(unsigned int vert_idx, const Vector<Scalar,Dim> &u)
{
    if(vert_idx >= vertex_displacements_.size())
        throw PhysikaException("Vertex index out of range.");
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
    if(vert_idx >= vertex_displacements_.size())
        throw PhysikaException("Vertex index out of range.");
    if(simulation_mesh_ == NULL)
        throw PhysikaException("Simulation mesh not set.");
    return simulation_mesh_->vertPos(vert_idx);
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> FEMBase<Scalar,Dim>::vertexCurrentPosition(unsigned int vert_idx) const
{
    if(vert_idx >= vertex_displacements_.size())
        throw PhysikaException("Vertex index out of range.");
    if(simulation_mesh_ == NULL)
        throw PhysikaException("Simulation mesh not set.");
    return simulation_mesh_->vertPos(vert_idx) + vertex_displacements_[vert_idx];  
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> FEMBase<Scalar,Dim>::vertexVelocity(unsigned int vert_idx) const
{
    if(vert_idx >= vertex_velocities_.size())
        throw PhysikaException("Vertex index out of range.");
    return vertex_velocities_[vert_idx];
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setVertexVelocity(unsigned int vert_idx, const Vector<Scalar,Dim> &v)
{
    if(vert_idx >= vertex_velocities_.size())
        throw PhysikaException("Vertex index out of range.");
    vertex_velocities_[vert_idx] = v;
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::resetVertexVelocity()
{
    for(unsigned int i = 0; i < vertex_velocities_.size(); ++i)
        vertex_velocities_[i] = Vector<Scalar,Dim>(0);
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> FEMBase<Scalar,Dim>::vertexExternalForce(unsigned int vert_idx) const
{
    if(vert_idx >= vertex_external_forces_.size())
        throw PhysikaException("Vertex index out of range.");
    return vertex_external_forces_[vert_idx];
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setVertexExternalForce(unsigned int vert_idx, const Vector<Scalar,Dim> &f)
{
    if(vert_idx >= vertex_external_forces_.size())
        throw PhysikaException("Vertex index out of range.");
    vertex_external_forces_[vert_idx] = f;
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::resetVertexExternalForce()
{
    for(unsigned int i = 0; i < vertex_external_forces_.size(); ++i)
        vertex_external_forces_[i] = Vector<Scalar,Dim>(0);
}

template <typename Scalar, int Dim>
unsigned int FEMBase<Scalar,Dim>::densityNum() const
{
    return material_density_.size();
}
    
template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setHomogeneousDensity(Scalar density)
{
    material_density_.clear();
    material_density_.push_back(density);
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setRegionWiseDensity(const std::vector<Scalar> &density)
{
    unsigned int region_num = this->simulation_mesh_->regionNum();
    if(density.size() < region_num)
        throw PhysikaException("Size of densities must be no less than the number of simulation mesh regions.");
    material_density_ = density;
}
 
template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setElementWiseDensity(const std::vector<Scalar> &density)
{
    unsigned int ele_num = this->simulation_mesh_->eleNum();
    if(density.size() < ele_num)
        throw PhysikaException("Size of densities must be no less than the number of simulation mesh elements.");
    material_density_ = density;
}
    
template <typename Scalar, int Dim>
Scalar FEMBase<Scalar,Dim>::elementDensity(unsigned int ele_idx) const
{
    unsigned int ele_num = this->simulation_mesh_->eleNum();
    unsigned int region_num = this->simulation_mesh_->regionNum();
    if(ele_idx >= ele_num)
        throw PhysikaException("Element index out of range.");
    unsigned int density_num = material_density_.size();
    if(density_num == 0)
        throw PhysikaException("Density not set.");
    else if(density_num == 1)
        return material_density_[0];
    else if(density_num == region_num)
    {
        int region_idx = this->simulation_mesh_->eleRegionIndex(ele_idx);
        if(region_idx==-1)
            throw PhysikaException("Element doesn't belong to any region, can't find its density in region-wise data.");
        else
            return material_density_[region_idx];
    }
    else if(density_num == ele_num)
        return material_density_[ele_idx];
    else
        PHYSIKA_ERROR("Invalid density number.");
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::applyVertexExternalForce()
{
    unsigned int vert_num = numSimVertices();
    Scalar dt = computeTimeStep();
    for(unsigned int i = 0; i < vert_num; ++i)
        vertex_velocities_[i] += vertex_external_forces_[i]/lumped_vertex_mass_[i]*dt;
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::synchronizeDataWithSimulationMesh()
{
    PHYSIKA_ASSERT(simulation_mesh_);
    unsigned int vert_num = simulation_mesh_->vertNum();
    vertex_displacements_.resize(vert_num);
    vertex_velocities_.resize(vert_num);
    vertex_external_forces_.resize(vert_num);
    lumped_vertex_mass_.resize(vert_num);
}

//explicit instantiations
template class FEMBase<float,2>;
template class FEMBase<float,3>;
template class FEMBase<double,2>;
template class FEMBase<double,3>;

}  //end of namespace Physika
