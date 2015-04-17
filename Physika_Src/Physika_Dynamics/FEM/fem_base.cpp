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
    :DriverBase<Scalar>(),
     cfl_num_(0.5),sound_speed_(340.0),gravity_(9.8)
{
}

template <typename Scalar, int Dim>
FEMBase<Scalar,Dim>::FEMBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :DriverBase<Scalar>(start_frame,end_frame,frame_rate,max_dt,write_to_file),
     cfl_num_(0.5),sound_speed_(340.0),gravity_(9.8)
{
}

template <typename Scalar, int Dim>
FEMBase<Scalar,Dim>::~FEMBase()
{
    clearSimulationMesh();
}

template <typename Scalar, int Dim>
Scalar FEMBase<Scalar,Dim>::computeTimeStep()
{
    //dt = cfl_num*L/(c+v_max), where L is the minimum characteristic length of element 
    //we approximate the chracteristic length of each element as square root of element volume in 2D and
    //cube root of element volume in 3D
    //for each object we compute a dt, and the global dt is the minimum of all objects
    Scalar dt = this->max_dt_;
    unsigned int obj_num = this->objectNum();
    for(unsigned int obj_idx = 0; obj_idx < obj_num; ++obj_num)
    {
        Scalar chracteristic_length = minElementCharacteristicLength(obj_idx);
        Scalar max_vel = maxVertexVelocityNorm(obj_idx);
        Scalar obj_dt = (cfl_num_*chracteristic_length)/(sound_speed_+max_vel);
        dt = dt < obj_dt ? dt : obj_dt;
    }
    this->dt_ = dt;
    return this->dt_;
}

template <typename Scalar, int Dim>
unsigned int FEMBase<Scalar,Dim>::objectNum() const
{
    return simulation_mesh_.size();
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::addObject(const VolumetricMesh<Scalar,Dim> &mesh, typename FEMBase<Scalar,Dim>::MassMatrixType mass_matrix_type)
{
    VolumetricMesh<Scalar,Dim> *vol_mesh = mesh.clone();
    simulation_mesh_.push_back(vol_mesh);
    if(mass_matrix_type != CONSISTENT_MASS && mass_matrix_type != LUMPED_MASS)
        throw PhysikaException("Unknown mass matrix type!");
    mass_matrix_type_.push_back(mass_matrix_type);
    //reserve space and initialize data associated with the object
    appendDataWithObject();
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::removeObject(unsigned int object_idx)
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    typename std::vector<VolumetricMesh<Scalar,Dim>*>::iterator iter1 = simulation_mesh_.begin() + object_idx;
    simulation_mesh_.erase(iter1);
    typename std::vector<MassMatrixType>::iterator iter2 = mass_matrix_type_.begin() + object_idx;
    mass_matrix_type_.erase(iter2);
    removeDataWithObject(object_idx);
}

template <typename Scalar, int Dim>
const VolumetricMesh<Scalar,Dim>& FEMBase<Scalar,Dim>::simulationMesh(unsigned int object_idx) const
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    PHYSIKA_ASSERT(simulation_mesh_[object_idx]);
    return *simulation_mesh_[object_idx];
}

template <typename Scalar, int Dim>
VolumetricMesh<Scalar,Dim>& FEMBase<Scalar,Dim>::simulationMesh(unsigned int object_idx)
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    PHYSIKA_ASSERT(simulation_mesh_[object_idx]);
    return *simulation_mesh_[object_idx];
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
unsigned int FEMBase<Scalar,Dim>::numSimVertices(unsigned int object_idx) const
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    PHYSIKA_ASSERT(simulation_mesh_[object_idx]);
    return simulation_mesh_[object_idx]->vertNum();
}

template <typename Scalar, int Dim>
unsigned int FEMBase<Scalar,Dim>::numSimElements(unsigned int object_idx) const
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    PHYSIKA_ASSERT(simulation_mesh_[object_idx]);
    return simulation_mesh_[object_idx]->eleNum();
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> FEMBase<Scalar,Dim>::vertexDisplacement(unsigned int object_idx, unsigned int vert_idx) const
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    if(vert_idx >= vertex_displacements_[object_idx].size())
        throw PhysikaException("Vertex index out of range!");
    return vertex_displacements_[object_idx][vert_idx];
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setVertexDisplacement(unsigned int object_idx, unsigned int vert_idx, const Vector<Scalar,Dim> &u)
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    if(vert_idx >= vertex_displacements_[object_idx].size())
        throw PhysikaException("Vertex index out of range!");
    vertex_displacements_[object_idx][vert_idx] = u;
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::resetVertexDisplacement(unsigned int object_idx)
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    for(unsigned int i = 0; i < vertex_displacements_[object_idx].size(); ++i)
        vertex_displacements_[object_idx][i] = Vector<Scalar,Dim>(0);
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> FEMBase<Scalar,Dim>::vertexRestPosition(unsigned int object_idx, unsigned int vert_idx) const
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    if(vert_idx >= vertex_displacements_[object_idx].size())
        throw PhysikaException("Vertex index out of range!");
    PHYSIKA_ASSERT(simulation_mesh_[object_idx]);
    return simulation_mesh_[object_idx]->vertPos(vert_idx);
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> FEMBase<Scalar,Dim>::vertexCurrentPosition(unsigned int object_idx, unsigned int vert_idx) const
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    if(vert_idx >= vertex_displacements_[object_idx].size())
        throw PhysikaException("Vertex index out of range!");
    PHYSIKA_ASSERT(simulation_mesh_[object_idx]);
    return simulation_mesh_[object_idx]->vertPos(vert_idx) + vertex_displacements_[object_idx][vert_idx];  
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> FEMBase<Scalar,Dim>::vertexVelocity(unsigned int object_idx, unsigned int vert_idx) const
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    if(vert_idx >= vertex_velocities_[object_idx].size())
        throw PhysikaException("Vertex index out of range!");
    return vertex_velocities_[object_idx][vert_idx];
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setVertexVelocity(unsigned int object_idx, unsigned int vert_idx, const Vector<Scalar,Dim> &v)
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    if(vert_idx >= vertex_velocities_[object_idx].size())
        throw PhysikaException("Vertex index out of range!");
    vertex_velocities_[object_idx][vert_idx] = v;
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::resetVertexVelocity(unsigned int object_idx)
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    for(unsigned int i = 0; i < vertex_velocities_[object_idx].size(); ++i)
        vertex_velocities_[object_idx][i] = Vector<Scalar,Dim>(0);
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> FEMBase<Scalar,Dim>::vertexExternalForce(unsigned int object_idx, unsigned int vert_idx) const
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    if(vert_idx >= vertex_external_forces_[object_idx].size())
        throw PhysikaException("Vertex index out of range!");
    return vertex_external_forces_[object_idx][vert_idx];
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setVertexExternalForce(unsigned int object_idx, unsigned int vert_idx, const Vector<Scalar,Dim> &f)
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    if(vert_idx >= vertex_external_forces_[object_idx].size())
        throw PhysikaException("Vertex index out of range!");
    vertex_external_forces_[object_idx][vert_idx] = f;
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::resetVertexExternalForce(unsigned int object_idx)
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    for(unsigned int i = 0; i < vertex_external_forces_[object_idx].size(); ++i)
        vertex_external_forces_[object_idx][i] = Vector<Scalar,Dim>(0);
}

template <typename Scalar, int Dim>
unsigned int FEMBase<Scalar,Dim>::densityNum(unsigned int object_idx) const
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    return material_density_[object_idx].size();
}
    
template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setHomogeneousDensity(unsigned int object_idx, Scalar density)
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    material_density_[object_idx].clear();
    material_density_[object_idx].push_back(density);
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setRegionWiseDensity(unsigned int object_idx, const std::vector<Scalar> &density)
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    unsigned int region_num = this->simulation_mesh_[object_idx]->regionNum();
    if(density.size() < region_num)
        throw PhysikaException("Size of densities doesn't match the number of simulation mesh regions!");
    material_density_[object_idx] = density;
}
 
template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setElementWiseDensity(unsigned int object_idx, const std::vector<Scalar> &density)
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    unsigned int ele_num = this->simulation_mesh_[object_idx]->eleNum();
    if(density.size() < ele_num)
        throw PhysikaException("Size of densities doesn't match the number of simulation mesh elements!");
    material_density_[object_idx] = density;
}
    
template <typename Scalar, int Dim>
Scalar FEMBase<Scalar,Dim>::elementDensity(unsigned int object_idx, unsigned int ele_idx) const
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    unsigned int ele_num = this->simulation_mesh_[object_idx]->eleNum();
    unsigned int region_num = this->simulation_mesh_[object_idx]->regionNum();
    if(ele_idx >= ele_num)
        throw PhysikaException("Element index out of range!");
    unsigned int density_num = material_density_[object_idx].size();
    if(density_num == 0)
        throw PhysikaException("Density not set!");
    else if(density_num == 1)
        return material_density_[object_idx][0];
    else if(density_num == region_num)
    {
        int region_idx = this->simulation_mesh_[object_idx]->eleRegionIndex(ele_idx);
        if(region_idx==-1)
            throw PhysikaException("Element doesn't belong to any region, can't find its density in region-wise data!");
        else
            return material_density_[object_idx][region_idx];
    }
    else if(density_num == ele_num)
        return material_density_[object_idx][ele_idx];
    else
        PHYSIKA_ERROR("Invalid density number!");
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::applyGravity(unsigned int object_idx, Scalar dt)
{
    PHYSIKA_ASSERT(object_idx < this->objectNum());
    unsigned int vert_num = numSimVertices(object_idx);
    Vector<Scalar,Dim> gravity_vec(0);
    gravity_vec[1] = -gravity_; //along negative y direction
    for(unsigned int i = 0; i < vert_num; ++i)
        vertex_velocities_[object_idx][i] += gravity_vec * dt;
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::appendDataWithObject()
{
    unsigned int last_obj_idx = this->objectNum() - 1;
    mass_matrix_.push_back(SparseMatrix<Scalar>());
    generateMassMatrix(last_obj_idx);
    unsigned int last_obj_vert_num = simulation_mesh_[last_obj_idx]->vertNum();
    std::vector<Vector<Scalar,Dim> > zero_vec(last_obj_vert_num,Vector<Scalar,Dim>(0));
    vertex_displacements_.push_back(zero_vec);
    vertex_velocities_.push_back(zero_vec);
    vertex_external_forces_.push_back(zero_vec);
    std::vector<Scalar> empty_material;
    material_density_.push_back(empty_material);
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::removeDataWithObject(unsigned int object_idx)
{
    PHYSIKA_ASSERT(object_idx < this->objectNum());
    typename std::vector<SparseMatrix<Scalar> >::iterator iter1 = mass_matrix_.begin() + object_idx;
    mass_matrix_.erase(iter1);
    typename std::vector<std::vector<Vector<Scalar,Dim> > >::iterator iter2 = vertex_displacements_.begin() + object_idx;
    vertex_displacements_.erase(iter2);
    typename std::vector<std::vector<Vector<Scalar,Dim> > >::iterator iter3 = vertex_velocities_.begin() + object_idx;
    vertex_velocities_.erase(iter3);
    typename std::vector<std::vector<Vector<Scalar,Dim> > >::iterator iter4 = vertex_external_forces_.begin() + object_idx;
    vertex_external_forces_.erase(iter4);
    typename std::vector<std::vector<Scalar> >::iterator iter5 = material_density_.begin() + object_idx;
    material_density_.erase(iter5);
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::generateMassMatrix(unsigned int object_idx)
{
    PHYSIKA_ASSERT(object_idx < this->objectNum());
    if(mass_matrix_type_[object_idx] == CONSISTENT_MASS)
    {
        typename VolumetricMeshMassGenerator<Scalar,Dim>::DensityOption density_option;
        if(material_density_[object_idx].size() == 1)
            VolumetricMeshMassGenerator<Scalar,Dim>::generateConsistentMass(*simulation_mesh_[object_idx],material_density_[object_idx][0],mass_matrix_[object_idx]);
        else if(material_density_[object_idx].size() == simulation_mesh_[object_idx]->eleNum())
        {
            density_option = VolumetricMeshMassGenerator<Scalar,Dim>::ELEMENT_WISE;
            VolumetricMeshMassGenerator<Scalar,Dim>::generateConsistentMass(*simulation_mesh_[object_idx],material_density_[object_idx],mass_matrix_[object_idx],density_option);
        }
        else if(material_density_[object_idx].size() == simulation_mesh_[object_idx]->regionNum())
        {
            density_option = VolumetricMeshMassGenerator<Scalar,Dim>::REGION_WISE;
            VolumetricMeshMassGenerator<Scalar,Dim>::generateConsistentMass(*simulation_mesh_[object_idx],material_density_[object_idx],mass_matrix_[object_idx],density_option);
        }
        else
            PHYSIKA_ERROR("Invalid density number!");
    }
    else if(mass_matrix_type_[object_idx] == LUMPED_MASS)
    {
        typename VolumetricMeshMassGenerator<Scalar,Dim>::DensityOption density_option;
        if(material_density_[object_idx].size() == 1)
            VolumetricMeshMassGenerator<Scalar,Dim>::generateLumpedMass(*simulation_mesh_[object_idx],material_density_[object_idx][0],mass_matrix_[object_idx]);
        else if(material_density_[object_idx].size() == simulation_mesh_[object_idx]->eleNum())
        {
            density_option = VolumetricMeshMassGenerator<Scalar,Dim>::ELEMENT_WISE;
            VolumetricMeshMassGenerator<Scalar,Dim>::generateLumpedMass(*simulation_mesh_[object_idx],material_density_[object_idx],mass_matrix_[object_idx],density_option);
        }
        else if(material_density_[object_idx].size() == simulation_mesh_[object_idx]->regionNum())
        {
            density_option = VolumetricMeshMassGenerator<Scalar,Dim>::REGION_WISE;
            VolumetricMeshMassGenerator<Scalar,Dim>::generateLumpedMass(*simulation_mesh_[object_idx],material_density_[object_idx],mass_matrix_[object_idx],density_option);
        }
        else
            PHYSIKA_ERROR("Invalid density number!");
    }
    else
        throw PhysikaException("Unknown mass matrix type!");
}

template <typename Scalar, int Dim>
Scalar FEMBase<Scalar,Dim>::maxVertexVelocityNorm(unsigned int object_idx) const
{
    PHYSIKA_ASSERT(object_idx < this->objectNum());
    Scalar max_norm_sqr = 0;
    for(unsigned int i = 0; i < vertex_velocities_[object_idx].size(); ++i)
    {
        Scalar vel_norm_sqr = vertex_velocities_[object_idx][i].normSquared();
        if(vel_norm_sqr > max_norm_sqr)
            max_norm_sqr = vel_norm_sqr;
    }
    return sqrt(max_norm_sqr);
}

template <typename Scalar, int Dim>
Scalar FEMBase<Scalar,Dim>::minElementCharacteristicLength(unsigned int object_idx) const
{
    PHYSIKA_ASSERT(object_idx < this->objectNum());
    PHYSIKA_ASSERT(simulation_mesh_[object_idx]);
    unsigned int ele_num = simulation_mesh_[object_idx]->eleNum();
    Scalar min_length = (std::numeric_limits<Scalar>::max)();
    std::vector<Vector<Scalar,2> > ele_vert_pos_trait_2d;
    std::vector<Vector<Scalar,3> > ele_vert_pos_trait_3d;
    VolumetricMeshInternal::ElementType ele_type = simulation_mesh_[object_idx]->elementType();
    for(unsigned int ele_idx = 0; ele_idx < ele_num; ++ele_idx)
    {
        unsigned int ele_vert_num = simulation_mesh_[object_idx]->eleVertNum();
        ele_vert_pos_trait_2d.resize(ele_vert_num);
        ele_vert_pos_trait_3d.resize(ele_vert_num);
        for(unsigned int local_vert_idx = 0; local_vert_idx < ele_vert_num; ++local_vert_idx)
        {
            unsigned int global_vert_idx = simulation_mesh_[object_idx]->eleVertIndex(ele_idx,local_vert_idx);
            Vector<Scalar,Dim> vert_pos = vertexCurrentPosition(object_idx,global_vert_idx);
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

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::clearSimulationMesh()
{
    for(unsigned int i = 0; i < simulation_mesh_.size(); ++i)
        if(simulation_mesh_[i])
            delete simulation_mesh_[i];
}

//explicit instantiations
template class FEMBase<float,2>;
template class FEMBase<float,3>;
template class FEMBase<double,2>;
template class FEMBase<double,3>;

}  //end of namespace Physika
