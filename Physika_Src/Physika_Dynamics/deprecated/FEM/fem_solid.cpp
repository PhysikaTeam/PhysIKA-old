/*
 * @file fem_isotropic_hyperelastic_solid.cpp 
 * @Brief FEM driver for isotropic hyperelastic solids, not necessarily homogeneous.
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

#include <limits>
#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Matrices/sparse_matrix_iterator.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh_internal.h"
#include "Physika_Dynamics/Constitutive_Models/constitutive_model.h"
#include "Physika_Dynamics/Collidable_Objects/collidable_object.h"
#include "Physika_Dynamics/FEM/FEM_Solid_Force_Model/fem_solid_force_model.h"
#include "Physika_Dynamics/FEM/FEM_Solid_Force_Model/tri_tet_mesh_fem_solid_force_model.h"
#include "Physika_Dynamics/FEM/FEM_Solid_Force_Model/quad_cubic_mesh_fem_solid_force_model.h"
#include "Physika_Dynamics/FEM/FEM_Plugins/fem_solid_plugin_base.h"
#include "Physika_Dynamics/FEM/fem_solid.h"

namespace Physika{

template <typename Scalar, int Dim>
FEMSolid<Scalar,Dim>::FEMSolid()
    :FEMBase<Scalar,Dim>(),integration_method_(FORWARD_EULER)
{
}

template <typename Scalar, int Dim>
FEMSolid<Scalar,Dim>::FEMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :FEMBase<Scalar,Dim>(start_frame,end_frame,frame_rate,max_dt,write_to_file),
     integration_method_(FORWARD_EULER)
{
}

template <typename Scalar, int Dim>
FEMSolid<Scalar,Dim>::~FEMSolid()
{
    clearAllMaterials();
    clearAllFEMSolidForceModels();
    clearKinematicObjects();
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::initConfiguration(const std::string &file_name)
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::printConfigFileFormat()
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::initSimulationData()
{//DO NOTHING FOR NOW
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::advanceStep(Scalar dt)
{
    //plugin operation, begin time step
    FEMSolidPluginBase<Scalar,Dim> *plugin = NULL;
    for(unsigned int i = 0; i < this->plugins_.size(); ++i)
    {
        plugin = dynamic_cast<FEMSolidPluginBase<Scalar,Dim>*>(this->plugins_[i]);
        if(plugin)
            plugin->onBeginTimeStep(this->time_,dt);
    }

    switch(this->integration_method_)
    {
    case FORWARD_EULER:
        advanceStepForwardEuler(dt);
        break;
    case BACKWARD_EULER:
        advanceStepBackwardEuler(dt);
        break;
    default:
    {
        std::string method_name = timeSteppingMethodName(this->integration_method_);
        throw PhysikaException(method_name+std::string("integration not implemented!"));
        break;
    }
    }
    this->time_ += dt;

    //plugin operation, end time step
    for(unsigned int i = 0; i < this->plugins_.size(); ++i)
    {
        plugin = dynamic_cast<FEMSolidPluginBase<Scalar,Dim>*>(this->plugins_[i]);
        if(plugin)
            plugin->onEndTimeStep(this->time_,dt);
    }
}

template <typename Scalar, int Dim>
bool FEMSolid<Scalar,Dim>::withRestartSupport() const
{
    return false;
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::write(const std::string &file_name)
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::read(const std::string &file_name)
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::addPlugin(DriverPluginBase<Scalar> *plugin)
{
    if(plugin == NULL)
    {
        std::cerr<<"Warning: NULL plugin provided, operation ignored!\n";
        return;
    }
    if(dynamic_cast<FEMSolidPluginBase<Scalar,Dim>*>(plugin) == NULL)
    {
        std::cerr<<"Warning: Wrong type of plugin provide, operation ignored!\n";
        return;
    }
    plugin->setDriver(this);
    this->plugins_.push_back(plugin);
}

template <typename Scalar, int Dim>
unsigned int FEMSolid<Scalar,Dim>::materialNum(unsigned int object_idx) const
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    return constitutive_model_[object_idx].size();
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::setHomogeneousMaterial(unsigned int object_idx, const ConstitutiveModel<Scalar,Dim> &material)
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    clearMaterial(object_idx);
    addMaterial(object_idx,material);
    //create fem force model
    createFEMSolidForceModel(object_idx);
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::setRegionWiseMaterial(unsigned int object_idx, const std::vector<ConstitutiveModel<Scalar,Dim>*> &materials)
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    unsigned int region_num = this->simulation_mesh_[object_idx]->regionNum();
    if(materials.size() < region_num)
        throw PhysikaException("Size of materials doesn't match the number of simulation mesh regions.");
    clearMaterial(object_idx);
    for(unsigned int i = 0; i < region_num; ++i)
        addMaterial(object_idx,*materials[i]);
    //create fem force model
    createFEMSolidForceModel(object_idx);
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::setElementWiseMaterial(unsigned int object_idx, const std::vector<ConstitutiveModel<Scalar,Dim>*> &materials)
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    unsigned int ele_num = this->simulation_mesh_[object_idx]->eleNum();
    if(materials.size() < ele_num)
        throw PhysikaException("Size of materials doesn't match the number of simulation mesh elements.");
    clearMaterial(object_idx);
    for(unsigned int i = 0; i < ele_num; ++i)
        addMaterial(object_idx,*materials[i]);
    //create fem force model
    createFEMSolidForceModel(object_idx);
}

template <typename Scalar, int Dim>
const ConstitutiveModel<Scalar,Dim>& FEMSolid<Scalar,Dim>::elementMaterial(unsigned int object_idx, unsigned int ele_idx) const
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    unsigned int ele_num = this->simulation_mesh_[object_idx]->eleNum();
    unsigned int region_num = this->simulation_mesh_[object_idx]->regionNum();
    if(ele_idx >= ele_num)
        throw PhysikaException("Element index out of range.");
    unsigned int material_num = constitutive_model_[object_idx].size();
    if(material_num == 0) //constitutive model not set
        throw PhysikaException("Element constitutive model not set.");
    else if(material_num == 1)//homogeneous material
        return *constitutive_model_[object_idx][0];
    else if(material_num == region_num)//region-wise material
    {
        int region_idx = this->simulation_mesh_[object_idx]->eleRegionIndex(ele_idx);
        if(region_idx==-1)
            throw PhysikaException("Element doesn't belong to any region, can't find its constitutive model in region-wise data.");
        else
            return *constitutive_model_[object_idx][region_idx];
    }
    else if(material_num == ele_num) //element-wise material
        return *constitutive_model_[object_idx][ele_idx];
    else
        PHYSIKA_ERROR("Invalid material number.");
}

template <typename Scalar, int Dim>
ConstitutiveModel<Scalar,Dim>& FEMSolid<Scalar,Dim>::elementMaterial(unsigned int object_idx, unsigned int ele_idx)
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    unsigned int ele_num = this->simulation_mesh_[object_idx]->eleNum();
    unsigned int region_num = this->simulation_mesh_[object_idx]->regionNum();
    if(ele_idx >= ele_num)
        throw PhysikaException("Element index out of range.");
    unsigned int material_num = constitutive_model_[object_idx].size();
    if(material_num == 0) //constitutive model not set
        throw PhysikaException("Element constitutive model not set.");
    else if(material_num == 1)//homogeneous material
        return *constitutive_model_[object_idx][0];
    else if(material_num == region_num)//region-wise material
    {
        int region_idx = this->simulation_mesh_[object_idx]->eleRegionIndex(ele_idx);
        if(region_idx==-1)
            throw PhysikaException("Element doesn't belong to any region, can't find its constitutive model in region-wise data.");
        else
            return *constitutive_model_[object_idx][region_idx];
    }
    else if(material_num == ele_num) //element-wise material
        return *constitutive_model_[object_idx][ele_idx];
    else
        PHYSIKA_ERROR("Invalid material number.");
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::setTimeSteppingMethod(TimeSteppingMethod method)
{
    integration_method_ = method;
}

template <typename Scalar, int Dim>
unsigned int FEMSolid<Scalar,Dim>::kinematicObjectNum() const
{
    return collidable_objects_.size();
}
    
template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::addKinematicObject(const CollidableObject<Scalar,Dim> &object)
{
    CollidableObject<Scalar,Dim> *new_object = object.clone();
    collidable_objects_.push_back(new_object);
}
        
template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::removeKinematicObject(unsigned int object_idx)
{
    if(object_idx >= collidable_objects_.size())
        std::cerr<<"Warning: kinematic object index out of range, operation ignored!\n";
    else
    {
        typename std::vector<CollidableObject<Scalar,Dim>*>::iterator iter = collidable_objects_.begin() + object_idx;
        collidable_objects_.erase(iter);
    }
}
        
template <typename Scalar, int Dim>
const CollidableObject<Scalar,Dim>& FEMSolid<Scalar,Dim>::kinematicObject(unsigned int object_idx) const
{
    if(object_idx >= collidable_objects_.size())
        throw PhysikaException("kinematic object index out of range!");
    PHYSIKA_ASSERT(collidable_objects_[object_idx]);
    return *collidable_objects_[object_idx];
}
        
template <typename Scalar, int Dim>
CollidableObject<Scalar,Dim>& FEMSolid<Scalar,Dim>::kinematicObject(unsigned int object_idx)
{
    if(object_idx >= collidable_objects_.size())
        throw PhysikaException("kinematic object index out of range!");
    PHYSIKA_ASSERT(collidable_objects_[object_idx]);
    return *collidable_objects_[object_idx];
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::setDirichletVertex(unsigned int object_idx, unsigned int vert_idx)
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    if(vert_idx >= this->numSimVertices(object_idx))
        throw PhysikaException("Vertex index out of range!");
    is_dirichlet_vertex_[object_idx][vert_idx] = 0x01;
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::setDirichletVertices(unsigned int object_idx, const std::vector<unsigned int> &vert_idx)
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("Object index out of range!");
    unsigned int invalid_vert = 0;
    for(unsigned int i = 0; i < vert_idx.size(); ++i)
    {
        if(vert_idx[i] >= this->numSimVertices(object_idx))
            ++invalid_vert;
        else
            is_dirichlet_vertex_[object_idx][vert_idx[i]] = 0x01;
    }
    if(invalid_vert >0)
        std::cerr<<"Warning: "<<invalid_vert<<" invalid vertex index are ignored!\n";
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::applyGravity(unsigned int object_idx, Scalar dt)
{
    PHYSIKA_ASSERT(object_idx < this->objectNum());
    unsigned int vert_num = this->numSimVertices(object_idx);
    Vector<Scalar,Dim> gravity_vec(0);
    gravity_vec[1] = -(this->gravity_); //along negative y direction
    for(unsigned int i = 0; i < vert_num; ++i)
    {
        if(is_dirichlet_vertex_[object_idx][i] == 0x00)
            this->vertex_velocities_[object_idx][i] += gravity_vec * dt;
    }
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::appendDataWithObject()
{
    FEMBase<Scalar,Dim>::appendDataWithObject();
    std::vector<ConstitutiveModel<Scalar,Dim>*> empty_material;
    constitutive_model_.push_back(empty_material);
    force_model_.push_back(NULL);
    unsigned int last_obj_idx = this->objectNum() - 1;
    unsigned int last_obj_vert_num = this->numSimVertices(last_obj_idx);
    std::vector<unsigned char> dirichlet(last_obj_vert_num,0x00);
    is_dirichlet_vertex_.push_back(dirichlet);
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::removeDataWithObject(unsigned int object_idx)
{
    PHYSIKA_ASSERT(object_idx < this->objectNum());
    FEMBase<Scalar,Dim>::removeDataWithObject(object_idx);
    typename std::vector<std::vector<ConstitutiveModel<Scalar,Dim>*> >::iterator iter1 = constitutive_model_.begin() + object_idx;
    constitutive_model_.erase(iter1);
    typename std::vector<FEMSolidForceModel<Scalar,Dim>*>::iterator iter2 = force_model_.begin() + object_idx;
    force_model_.erase(iter2);
    std::vector<std::vector<unsigned char> >::iterator iter3 = is_dirichlet_vertex_.begin() + object_idx;
    is_dirichlet_vertex_.erase(iter3);
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::clearAllMaterials()
{
    for(unsigned int i = 0 ; i < constitutive_model_.size(); ++i)
        for(unsigned int j = 0; j < constitutive_model_[i].size(); ++j)
            if(constitutive_model_[i][j])
                delete constitutive_model_[i][j];
    constitutive_model_.clear();
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::clearMaterial(unsigned int object_idx)
{
    PHYSIKA_ASSERT(object_idx < this->objectNum());
    for(unsigned int i = 0 ; i < constitutive_model_[object_idx].size(); ++i)
        if(constitutive_model_[object_idx][i])
            delete constitutive_model_[object_idx][i];
    constitutive_model_[object_idx].clear();
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::addMaterial(unsigned int object_idx, const ConstitutiveModel<Scalar,Dim> &material)
{
    PHYSIKA_ASSERT(object_idx < this->objectNum());
    ConstitutiveModel<Scalar,Dim> *single_material = material.clone();
    PHYSIKA_ASSERT(single_material);
    constitutive_model_[object_idx].push_back(single_material);
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::advanceStepForwardEuler(Scalar dt)
{
    unsigned int obj_num = this->objectNum();
    for(unsigned int obj_idx = 0; obj_idx < obj_num; ++obj_idx)
    {
        if(this->mass_matrix_type_[obj_idx] == FEMBase<Scalar,Dim>::CONSISTENT_MASS)
        {
            //using consistent matrix requires solving a linear system
            throw PhysikaException("Not implemented!");
        }
        else if(this->mass_matrix_type_[obj_idx] == FEMBase<Scalar,Dim>::LUMPED_MASS)
        {
            unsigned int vert_num = this->numSimVertices(obj_idx);
            //first compute internal forces
            std::vector<Vector<Scalar,Dim> > internal_force;
            PHYSIKA_ASSERT(force_model_[obj_idx]);
            std::vector<Vector<Scalar,Dim> > vertex_cur_pos(vert_num);
            for(unsigned int vert_idx = 0; vert_idx < vert_num; ++vert_idx)
                vertex_cur_pos[vert_idx] = this->vertexCurrentPosition(obj_idx,vert_idx);
            force_model_[obj_idx]->computeGlobalInternalForces(vertex_cur_pos,internal_force);
            //now apply internal forces and external forces
            SparseMatrixIterator<Scalar> mat_iter(this->mass_matrix_[obj_idx]);
            while(mat_iter)
            {
                unsigned int row = mat_iter.row();
                unsigned int col = mat_iter.col();
                PHYSIKA_ASSERT(row == col);
                Scalar vert_mass =mat_iter.value();
                if(is_dirichlet_vertex_[obj_idx][row] == 0x00) //apply forces if not a dirichlet vertex
                {
                    (this->vertex_velocities_[obj_idx])[row] += internal_force[row]/vert_mass*dt;//internal force
                    (this->vertex_velocities_[obj_idx])[row] += (this->vertex_external_forces_[obj_idx])[row]/vert_mass*dt;//external force
                }
                ++mat_iter;
            }
            //apply gravity
            this->applyGravity(obj_idx,dt);
            //update vertex positions with the new velocities
            for(unsigned int vert_idx = 0; vert_idx < vert_num; ++vert_idx)
                (this->vertex_displacements_[obj_idx])[vert_idx] += (this->vertex_velocities_[obj_idx])[vert_idx]*dt;
            //after update the vertex positions, resolve the contact with kinematic objects in scene
            resolveContactWithKinematicObjects(obj_idx);
        }
        else
            PHYSIKA_ERROR("Unknown mass matrix type!");
    }
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::advanceStepBackwardEuler(Scalar dt)
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::createFEMSolidForceModel(unsigned int object_idx)
{
    PHYSIKA_ASSERT(object_idx < this->objectNum());
    clearFEMSolidForceModel(object_idx);
    VolumetricMeshInternal::ElementType ele_type = (this->simulation_mesh_[object_idx])->elementType();
    switch(ele_type)
    {
    case VolumetricMeshInternal::TRI:
    case VolumetricMeshInternal::TET:
        force_model_[object_idx] = new TriTetMeshFEMSolidForceModel<Scalar,Dim>(*(this->simulation_mesh_[object_idx]),constitutive_model_[object_idx]);
        break;
    case VolumetricMeshInternal::QUAD:
    case VolumetricMeshInternal::CUBIC:
        force_model_[object_idx] = new QuadCubicMeshFEMSolidForceModel<Scalar,Dim>(*(this->simulation_mesh_[object_idx]),constitutive_model_[object_idx]);
        break;
    default:
        PHYSIKA_ERROR("Unsupported element type!");
    }
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::clearAllFEMSolidForceModels()
{
    for(unsigned int i = 0; i < force_model_.size(); ++i)
        if(force_model_[i])
            delete force_model_[i];
    force_model_.clear();
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::clearFEMSolidForceModel(unsigned int object_idx)
{
    PHYSIKA_ASSERT(object_idx < this->objectNum());
    if(force_model_[object_idx])
        delete force_model_[object_idx];
    force_model_[object_idx] = NULL;
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::clearKinematicObjects()
{
    //delete the kinematic objects
    for(unsigned int i = 0; i < collidable_objects_.size(); ++i)
        if(collidable_objects_[i])
            delete collidable_objects_[i];
    collidable_objects_.clear();
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::resolveContactWithKinematicObjects(unsigned int object_idx)
{
    //resolve contact with the kinematic objects in scene
    //detect collision between the mesh vertices and the closest collidable object
    //impulse will be applied on the colliding vertices and the vertices will be projected onto the surface of colliding object
    if(!(this->collidable_objects_).empty())
    {
        for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
        {
            for(unsigned int vert_idx = 0; vert_idx < this->numSimVertices(obj_idx); ++vert_idx)
            {
                if(is_dirichlet_vertex_[obj_idx][vert_idx])  //skip dirichlet vertices
                    continue;
                Vector<Scalar,Dim> vert_pos = this->vertexCurrentPosition(obj_idx,vert_idx);
                //get closest kinematic object idx
                Scalar closest_dist = (std::numeric_limits<Scalar>::max)();
                unsigned int closest_obj_idx = 0;
                for(unsigned int i = 0; i < this->collidable_objects_.size(); ++i)
                {
                    Scalar signed_distance = (this->collidable_objects_)[i]->signedDistance(vert_pos);
                    if(signed_distance < closest_dist)
                    {
                        closest_dist = signed_distance;
                        closest_obj_idx = i;
                    }
                }
                Vector<Scalar,Dim> vert_vel = this->vertexVelocity(obj_idx,vert_idx);
                Vector<Scalar,Dim> impulse(0);
                Scalar collide_threshold = this->collidable_objects_[closest_obj_idx]->collideThreshold();
                if(this->collidable_objects_[closest_obj_idx]->collide(vert_pos,vert_vel,impulse))
                {
                    //project the vertex onto the surface of the collidable object
                    Vector<Scalar,Dim> normal = this->collidable_objects_[closest_obj_idx]->normal(vert_pos);
                    vert_pos += (collide_threshold - closest_dist) * normal;
                    Vector<Scalar,Dim> vert_ref_pos = this->vertexRestPosition(obj_idx,vert_idx);
                    this->setVertexDisplacement(obj_idx,vert_idx,vert_pos-vert_ref_pos);
                    //apply the impulse to change vertex velocity
                    Vector<Scalar,Dim> vert_vel = this->vertexVelocity(obj_idx,vert_idx);
                    vert_vel += impulse;
                    this->setVertexVelocity(obj_idx,vert_idx,vert_vel);
                }
            }
        }
    }
}

//explicit instantiations
template class FEMSolid<float,2>;
template class FEMSolid<double,2>;
template class FEMSolid<float,3>;
template class FEMSolid<double,3>;

}  //end of namespace Physika
