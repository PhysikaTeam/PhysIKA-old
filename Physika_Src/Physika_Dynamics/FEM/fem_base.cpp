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

#include <limits>
#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_IO/Volumetric_Mesh_IO/volumetric_mesh_io.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh_internal.h"
#include "Physika_Geometry/Volumetric_Meshes/tri_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/quad_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/tet_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/cubic_mesh.h"
#include "Physika_Dynamics/Utilities/Volumetric_Mesh_Mass_Generator/volumetric_mesh_mass_generator.h"
#include "Physika_Dynamics/FEM/fem_base.h"

namespace Physika{

template <typename Scalar, int Dim>
FEMBase<Scalar,Dim>::FEMBase()
    :DriverBase<Scalar>(),simulation_mesh_(NULL),mass_matrix_type_(FEMBase<Scalar,Dim>::LUMPED_MASS),gravity_(9.8),cfl_num_(0.5),sound_speed_(340.0)
{
}

template <typename Scalar, int Dim>
FEMBase<Scalar,Dim>::FEMBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :DriverBase<Scalar>(start_frame,end_frame,frame_rate,max_dt,write_to_file),
     simulation_mesh_(NULL),mass_matrix_type_(FEMBase<Scalar,Dim>::LUMPED_MASS),gravity_(9.8),cfl_num_(0.5),sound_speed_(340.0)
{
}

template <typename Scalar, int Dim>
FEMBase<Scalar,Dim>::FEMBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file,
                             const VolumetricMesh<Scalar,Dim> &mesh)
    :DriverBase<Scalar>(start_frame,end_frame,frame_rate,max_dt,write_to_file),
     simulation_mesh_(NULL),mass_matrix_type_(FEMBase<Scalar,Dim>::LUMPED_MASS),gravity_(9.8),cfl_num_(0.5),sound_speed_(340.0)
{
    setSimulationMesh(mesh);
}

template <typename Scalar, int Dim>
FEMBase<Scalar,Dim>::FEMBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file,
                             const VolumetricMesh<Scalar,Dim> &mesh, typename FEMBase<Scalar,Dim>::MassMatrixType mass_matrix_type)
    :DriverBase<Scalar>(start_frame,end_frame,frame_rate,max_dt,write_to_file),
     simulation_mesh_(NULL),mass_matrix_type_(mass_matrix_type),gravity_(9.8),cfl_num_(0.5),sound_speed_(340.0)
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
Scalar FEMBase<Scalar,Dim>::computeTimeStep()
{
    //dt = cfl_num*L/(c+v_max), where L is the minimum characteristic length of element 
    //we approximate the chracteristic length of each element as square root of element volume in 2D and
    //cube root of element volume in 3D
    Scalar chracteristic_length = minElementCharacteristicLength();
    Scalar max_vel = maxVertexVelocityNorm();
    this->dt_ = (cfl_num_*chracteristic_length)/(sound_speed_+max_vel);
    this->dt_ = this->dt_ > this->max_dt_ ? this->max_dt_ : this->dt_;
    return this->dt_;
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
Scalar FEMBase<Scalar,Dim>::cflConstant() const
{
    return cfl_num_;
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setCFLConstant(Scalar cfl)
{
    if(cfl<0)
    {
        std::cerr<<"Warning: Invalid CFL constant specified, use default value (0.5) instead!\n";
        cfl_num_ = 0.5;
    }
    else
        cfl_num_ = cfl;
}

template <typename Scalar, int Dim>
Scalar FEMBase<Scalar,Dim>::soundSpeed() const
{
    return sound_speed_;
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setSoundSpeed(Scalar sound_speed)
{
    if(sound_speed<0)
    {
        std::cerr<<"Warning: Negative sound speed specified, use its absolute value instead!\n";
        sound_speed_ = -sound_speed;
    }
    else
        sound_speed_ = sound_speed;
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
        throw PhysikaException("FEM with non-uniform element type not implemented yet!");
        break;
    default:
        PHYSIKA_ERROR("Unknown element type!");
        break;
    }
    synchronizeDataWithSimulationMesh();
}

template <typename Scalar, int Dim>
const VolumetricMesh<Scalar,Dim>& FEMBase<Scalar,Dim>::simulationMesh() const
{
    if(simulation_mesh_==NULL)
        throw PhysikaException("Simulation mesh not set!");
    return *simulation_mesh_;
}

template <typename Scalar, int Dim>
VolumetricMesh<Scalar,Dim>& FEMBase<Scalar,Dim>::simulationMesh()
{
    if(simulation_mesh_==NULL)
        throw PhysikaException("Simulation mesh not set!");
    return *simulation_mesh_;
}

template <typename Scalar, int Dim>
unsigned int FEMBase<Scalar,Dim>::numSimVertices() const
{
    if(simulation_mesh_==NULL)
        throw PhysikaException("Simulation mesh not set!");
    return simulation_mesh_->vertNum();
}

template <typename Scalar, int Dim>
unsigned int FEMBase<Scalar,Dim>::numSimElements() const
{
    if(simulation_mesh_==NULL)
        throw PhysikaException("Simulation mesh not set!");
    return simulation_mesh_->eleNum();
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> FEMBase<Scalar,Dim>::vertexDisplacement(unsigned int vert_idx) const
{
    if(vert_idx >= vertex_displacements_.size())
        throw PhysikaException("Vertex index out of range!");
    return vertex_displacements_[vert_idx];
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setVertexDisplacement(unsigned int vert_idx, const Vector<Scalar,Dim> &u)
{
    if(vert_idx >= vertex_displacements_.size())
        throw PhysikaException("Vertex index out of range!");
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
        throw PhysikaException("Vertex index out of range!");
    if(simulation_mesh_ == NULL)
        throw PhysikaException("Simulation mesh not set!");
    return simulation_mesh_->vertPos(vert_idx);
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> FEMBase<Scalar,Dim>::vertexCurrentPosition(unsigned int vert_idx) const
{
    if(vert_idx >= vertex_displacements_.size())
        throw PhysikaException("Vertex index out of range!");
    if(simulation_mesh_ == NULL)
        throw PhysikaException("Simulation mesh not set!");
    return simulation_mesh_->vertPos(vert_idx) + vertex_displacements_[vert_idx];  
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> FEMBase<Scalar,Dim>::vertexVelocity(unsigned int vert_idx) const
{
    if(vert_idx >= vertex_velocities_.size())
        throw PhysikaException("Vertex index out of range!");
    return vertex_velocities_[vert_idx];
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setVertexVelocity(unsigned int vert_idx, const Vector<Scalar,Dim> &v)
{
    if(vert_idx >= vertex_velocities_.size())
        throw PhysikaException("Vertex index out of range!");
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
        throw PhysikaException("Vertex index out of range!");
    return vertex_external_forces_[vert_idx];
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setVertexExternalForce(unsigned int vert_idx, const Vector<Scalar,Dim> &f)
{
    if(vert_idx >= vertex_external_forces_.size())
        throw PhysikaException("Vertex index out of range!");
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
        throw PhysikaException("Size of densities must be no less than the number of simulation mesh regions!");
    material_density_ = density;
}
 
template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setElementWiseDensity(const std::vector<Scalar> &density)
{
    unsigned int ele_num = this->simulation_mesh_->eleNum();
    if(density.size() < ele_num)
        throw PhysikaException("Size of densities must be no less than the number of simulation mesh elements!");
    material_density_ = density;
}
    
template <typename Scalar, int Dim>
Scalar FEMBase<Scalar,Dim>::elementDensity(unsigned int ele_idx) const
{
    unsigned int ele_num = this->simulation_mesh_->eleNum();
    unsigned int region_num = this->simulation_mesh_->regionNum();
    if(ele_idx >= ele_num)
        throw PhysikaException("Element index out of range!");
    unsigned int density_num = material_density_.size();
    if(density_num == 0)
        throw PhysikaException("Density not set!");
    else if(density_num == 1)
        return material_density_[0];
    else if(density_num == region_num)
    {
        int region_idx = this->simulation_mesh_->eleRegionIndex(ele_idx);
        if(region_idx==-1)
            throw PhysikaException("Element doesn't belong to any region, can't find its density in region-wise data!");
        else
            return material_density_[region_idx];
    }
    else if(density_num == ele_num)
        return material_density_[ele_idx];
    else
        PHYSIKA_ERROR("Invalid density number!");
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::applyGravity(Scalar dt)
{
    unsigned int vert_num = numSimVertices();
    Vector<Scalar,Dim> gravity_vec(0);
    gravity_vec[1] = -gravity_; //along negative y direction
    for(unsigned int i = 0; i < vert_num; ++i)
        vertex_velocities_[i] += gravity_vec * dt;
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::synchronizeDataWithSimulationMesh()
{
    PHYSIKA_ASSERT(simulation_mesh_);
    unsigned int vert_num = simulation_mesh_->vertNum();
    vertex_displacements_.resize(vert_num);
    vertex_velocities_.resize(vert_num);
    vertex_external_forces_.resize(vert_num);
    generateMassMatrix();
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::generateMassMatrix()
{
    if(mass_matrix_type_ == CONSISTENT_MASS)
    {
        typename VolumetricMeshMassGenerator<Scalar,Dim>::DensityOption density_option;
        if(material_density_.size() == 1)
            VolumetricMeshMassGenerator<Scalar,Dim>::generateConsistentMass(*simulation_mesh_,material_density_[0],mass_matrix_);
        else if(material_density_.size() == simulation_mesh_->eleNum())
        {
            density_option = VolumetricMeshMassGenerator<Scalar,Dim>::ELEMENT_WISE;
            VolumetricMeshMassGenerator<Scalar,Dim>::generateConsistentMass(*simulation_mesh_,material_density_,mass_matrix_,density_option);
        }
        else if(material_density_.size() == simulation_mesh_->regionNum())
        {
            density_option = VolumetricMeshMassGenerator<Scalar,Dim>::REGION_WISE;
            VolumetricMeshMassGenerator<Scalar,Dim>::generateConsistentMass(*simulation_mesh_,material_density_,mass_matrix_,density_option);
        }
        else
            PHYSIKA_ERROR("Invalid density number!");
    }
    else if(mass_matrix_type_ == LUMPED_MASS)
    {
        typename VolumetricMeshMassGenerator<Scalar,Dim>::DensityOption density_option;
        if(material_density_.size() == 1)
            VolumetricMeshMassGenerator<Scalar,Dim>::generateLumpedMass(*simulation_mesh_,material_density_[0],mass_matrix_);
        else if(material_density_.size() == simulation_mesh_->eleNum())
        {
            density_option = VolumetricMeshMassGenerator<Scalar,Dim>::ELEMENT_WISE;
            VolumetricMeshMassGenerator<Scalar,Dim>::generateLumpedMass(*simulation_mesh_,material_density_,mass_matrix_,density_option);
        }
        else if(material_density_.size() == simulation_mesh_->regionNum())
        {
            density_option = VolumetricMeshMassGenerator<Scalar,Dim>::REGION_WISE;
            VolumetricMeshMassGenerator<Scalar,Dim>::generateLumpedMass(*simulation_mesh_,material_density_,mass_matrix_,density_option);
        }
        else
            PHYSIKA_ERROR("Invalid density number!");
    }
    else
        throw PhysikaException("Unknown mass matrix type!");
}

template <typename Scalar, int Dim>
Scalar FEMBase<Scalar,Dim>::maxVertexVelocityNorm() const
{
    Scalar max_norm_sqr = 0;
    for(unsigned int i = 0; i < vertex_velocities_.size(); ++i)
    {
        Scalar vel_norm_sqr = vertex_velocities_[i].normSquared();
        if(vel_norm_sqr > max_norm_sqr)
            max_norm_sqr = vel_norm_sqr;
    }
    return sqrt(max_norm_sqr);
}

template <typename Scalar, int Dim>
Scalar FEMBase<Scalar,Dim>::minElementCharacteristicLength() const
{
    PHYSIKA_ASSERT(simulation_mesh_);
    unsigned int ele_num = simulation_mesh_->eleNum();
    Scalar min_length = (std::numeric_limits<Scalar>::max)();
    std::vector<Vector<Scalar,2> > ele_vert_pos_trait_2d;
    std::vector<Vector<Scalar,3> > ele_vert_pos_trait_3d;
    VolumetricMeshInternal::ElementType ele_type = simulation_mesh_->elementType();
    for(unsigned int ele_idx = 0; ele_idx < ele_num; ++ele_idx)
    {
        unsigned int ele_vert_num = simulation_mesh_->eleVertNum();
        ele_vert_pos_trait_2d.resize(ele_vert_num);
        ele_vert_pos_trait_3d.resize(ele_vert_num);
        for(unsigned int local_vert_idx = 0; local_vert_idx < ele_vert_num; ++local_vert_idx)
        {
            unsigned int global_vert_idx = simulation_mesh_->eleVertIndex(ele_idx,local_vert_idx);
            Vector<Scalar,Dim> vert_pos = vertexCurrentPosition(global_vert_idx);
            for(unsigned int i = 0; i < 2; ++i)
            {
                ele_vert_pos_trait_2d[local_vert_idx][i] = vert_pos[i];
                ele_vert_pos_trait_3d[local_vert_idx][i] = vert_pos[i];
            }
            ele_vert_pos_trait_3d[local_vert_idx][2] = vert_pos[2];
        }
        //approximate the characteristic length of the element as the square/cube root of element volume
        Scalar ele_length = min_length;
        //assume vertices of elements are in correct order
        switch(ele_type)
        {
        case VolumetricMeshInternal::TRI:
        {
            Vector<Scalar,2> b_minus_a = ele_vert_pos_trait_2d[1] - ele_vert_pos_trait_2d[0];
            Vector<Scalar,2> c_minus_a = ele_vert_pos_trait_2d[2] - ele_vert_pos_trait_2d[0]; 
            ele_length = abs(b_minus_a.cross(c_minus_a))/2.0;
            ele_length = sqrt(ele_length);
            break;
        }
        case VolumetricMeshInternal::TET:
        {
            Vector<Scalar,3> a_minus_d = ele_vert_pos_trait_3d[0] - ele_vert_pos_trait_3d[3];
            Vector<Scalar,3> b_minus_d = ele_vert_pos_trait_3d[1] - ele_vert_pos_trait_3d[3];
            Vector<Scalar,3> c_minus_d = ele_vert_pos_trait_3d[2] - ele_vert_pos_trait_3d[3]; 
            ele_length = 1.0/6*abs(a_minus_d.dot(b_minus_d.cross(c_minus_d)));
            ele_length = cbrt(ele_length);
            break;
        }
        case VolumetricMeshInternal::QUAD:
        {
            Vector<Scalar,2> a_minus_d = ele_vert_pos_trait_2d[0] - ele_vert_pos_trait_2d[3];
            Vector<Scalar,2> b_minus_d = ele_vert_pos_trait_2d[1] - ele_vert_pos_trait_2d[3];
            Vector<Scalar,2> c_minus_d = ele_vert_pos_trait_2d[2] - ele_vert_pos_trait_2d[3]; 
            ele_length = 1.0/2*abs((b_minus_d.cross(a_minus_d)) + (b_minus_d.cross(c_minus_d)));
            ele_length = sqrt(ele_length);
            break;
        }
        case VolumetricMeshInternal::CUBIC:
        {
            Vector<Scalar,3> fir_minus_0 = ele_vert_pos_trait_3d[1] - ele_vert_pos_trait_3d[0];
            Vector<Scalar,3> thi_minus_0 = ele_vert_pos_trait_3d[3] - ele_vert_pos_trait_3d[0];
            Vector<Scalar,3> fou_minus_0 = ele_vert_pos_trait_3d[4] - ele_vert_pos_trait_3d[0]; 
            ele_length = 1.0 * (fir_minus_0.norm() * thi_minus_0.norm() * fou_minus_0.norm() ) ;   
            ele_length = cbrt(ele_length);
            break;
        }
        default:
            PHYSIKA_ERROR("Unsupported element type!");
            break;
        }
        min_length = ele_length < min_length ? ele_length : min_length;
    }
    return min_length;
}

//explicit instantiations
template class FEMBase<float,2>;
template class FEMBase<float,3>;
template class FEMBase<double,2>;
template class FEMBase<double,3>;

}  //end of namespace Physika
