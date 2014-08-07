/*
 * @file fem_isotropic_hyperelastic_solid.cpp 
 * @Brief FEM driver for isotropic hyperelastic solids, not necessarily homogeneous.
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
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_Dynamics/Constitutive_Models/constitutive_model.h"
#include "Physika_Dynamics/FEM/fem_solid.h"

namespace Physika{

template <typename Scalar, int Dim>
FEMSolid<Scalar,Dim>::FEMSolid()
    :FEMBase<Scalar,Dim>()
{
}

template <typename Scalar, int Dim>
FEMSolid<Scalar,Dim>::FEMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :FEMBase<Scalar,Dim>(start_frame,end_frame,frame_rate,max_dt,write_to_file)
{
}

template <typename Scalar, int Dim>
FEMSolid<Scalar,Dim>::FEMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file,
                               const VolumetricMesh<Scalar,Dim> &mesh)
    :FEMBase<Scalar,Dim>(start_frame,end_frame,frame_rate,max_dt,write_to_file,mesh)
{
}

template <typename Scalar, int Dim>
FEMSolid<Scalar,Dim>::~FEMSolid()
{
    for(unsigned int i = 0; i < constitutive_model_.size(); ++i)
        delete constitutive_model_[i];
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::initConfiguration(const std::string &file_name)
{//TO DO
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::printConfigFileFormat()
{//TO DO
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::initSimulationData()
{//TO DO
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::advanceStep(Scalar dt)
{//TO DO
}

template <typename Scalar, int Dim>
Scalar FEMSolid<Scalar,Dim>::computeTimeStep()
{
//TO DO
    return 0;
}

template <typename Scalar, int Dim>
bool FEMSolid<Scalar,Dim>::withRestartSupport() const
{
    return false;
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::write(const std::string &file_name)
{//TO DO
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::read(const std::string &file_name)
{//TO DO
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::addPlugin(DriverPluginBase<Scalar> *plugin)
{//TO DO
}

template <typename Scalar, int Dim>
unsigned int FEMSolid<Scalar,Dim>::materialNum() const
{
    return constitutive_model_.size();
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::setHomogeneousMaterial(const ConstitutiveModel<Scalar,Dim> &material)
{
    clearMaterial();
    addMaterial(material);
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::setRegionWiseMaterial(const std::vector<ConstitutiveModel<Scalar,Dim>*> &materials)
{
    unsigned int region_num = this->simulation_mesh_->regionNum();
    if(materials.size() < region_num)
    {
        std::cerr<<"Size of materials must be no less than the number of simulation mesh regions.\n";
        std::exit(EXIT_FAILURE);
    }
    clearMaterial();
    for(unsigned int i = 0; i < region_num; ++i)
        addMaterial(*materials[i]);
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::setElementWiseMaterial(const std::vector<ConstitutiveModel<Scalar,Dim>*> &materials)
{
    unsigned int ele_num = this->simulation_mesh_->eleNum();
    if(materials.size() < ele_num)
    {
        std::cerr<<"Size of materials must be no less than the number of simulation mesh elements.\n";
        std::exit(EXIT_FAILURE);
    }
    clearMaterial();
    for(unsigned int i = 0; i < ele_num; ++i)
        addMaterial(*materials[i]);
}

template <typename Scalar, int Dim>
const ConstitutiveModel<Scalar,Dim>* FEMSolid<Scalar,Dim>::elementMaterial(unsigned int ele_idx) const
{
    unsigned int ele_num = this->simulation_mesh_->eleNum();
    unsigned int region_num = this->simulation_mesh_->regionNum();
    if(ele_idx >= ele_num)
    {
        std::cerr<<"Element index out of range.\n";
        std::exit(EXIT_FAILURE);
    }
    unsigned int material_num = constitutive_model_.size();
    if(material_num == 0) //constitutive model not set
        return NULL;
    else if(material_num == 1)//homogeneous material
        return constitutive_model_[0];
    else if(material_num == region_num)//region-wise material
    {
        int region_idx = this->simulation_mesh_->eleRegionIndex(ele_idx);
        return (region_idx == -1)? NULL : constitutive_model_[region_idx];
    }
    else if(material_num == ele_num) //element-wise material
        return constitutive_model_[ele_idx];
    else
        PHYSIKA_ERROR("Invalid material number.");
}

template <typename Scalar, int Dim>
ConstitutiveModel<Scalar,Dim>* FEMSolid<Scalar,Dim>::elementMaterial(unsigned int ele_idx)
{
    unsigned int ele_num = this->simulation_mesh_->eleNum();
    unsigned int region_num = this->simulation_mesh_->regionNum();
    if(ele_idx >= ele_num)
    {
        std::cerr<<"Element index out of range.\n";
        std::exit(EXIT_FAILURE);
    }
    unsigned int material_num = constitutive_model_.size();
    if(material_num == 0) //constitutive model not set
        return NULL;
    else if(material_num == 1)//homogeneous material
        return constitutive_model_[0];
    else if(material_num == region_num)//region-wise material
    {
        int region_idx = this->simulation_mesh_->eleRegionIndex(ele_idx);
        return (region_idx == -1)? NULL : constitutive_model_[region_idx];
    }
    else if(material_num == ele_num) //element-wise material
        return constitutive_model_[ele_idx];
    else
        PHYSIKA_ERROR("Invalid material number.");
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::clearMaterial()
{
    for(unsigned int i = 0 ; i < constitutive_model_.size(); ++i)
        if(constitutive_model_[i])
            delete constitutive_model_[i];
    constitutive_model_.clear();
}

template <typename Scalar, int Dim>
void FEMSolid<Scalar,Dim>::addMaterial(const ConstitutiveModel<Scalar,Dim> &material)
{
    ConstitutiveModel<Scalar,Dim> *single_material = material.clone();
    PHYSIKA_ASSERT(single_material);
    constitutive_model_.push_back(single_material);
}

//explicit instantiations
template class FEMSolid<float,2>;
template class FEMSolid<double,2>;
template class FEMSolid<float,3>;
template class FEMSolid<double,3>;

}  //end of namespace Physika
