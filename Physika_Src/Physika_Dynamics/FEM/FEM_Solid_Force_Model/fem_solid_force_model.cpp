/*
 * @file fem_solid_force_model.cpp 
 * @Brief the "engine" for fem solid drivers.
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

#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_Dynamics/Constitutive_Models/constitutive_model.h"
#include "Physika_Dynamics/FEM/FEM_Solid_Force_Model/fem_solid_force_model.h"

namespace Physika{

template <typename Scalar, int Dim>
FEMSolidForceModel<Scalar,Dim>::FEMSolidForceModel(const VolumetricMesh<Scalar,Dim> &simulation_mesh, const std::vector<ConstitutiveModel<Scalar,Dim>*> &constitutive_model)
    :simulation_mesh_(simulation_mesh), constitutive_model_(constitutive_model)
{
}

template <typename Scalar, int Dim>    
FEMSolidForceModel<Scalar,Dim>::~FEMSolidForceModel()
{
}

template <typename Scalar, int Dim>    
const ConstitutiveModel<Scalar,Dim>& FEMSolidForceModel<Scalar,Dim>::elementMaterial(unsigned int ele_idx) const
{
    unsigned int ele_num = simulation_mesh_.eleNum();
    unsigned int region_num = simulation_mesh_.regionNum();
    if(ele_idx >= ele_num)
        throw PhysikaException("Element index out of range.");
    unsigned int material_num = constitutive_model_.size();
    if(material_num == 0) //constitutive model not set
        throw PhysikaException("Element constitutive model not set.");
    else if(material_num == 1)//homogeneous material
        return *constitutive_model_[0];
    else if(material_num == region_num)//region-wise material
    {
        int region_idx = simulation_mesh_.eleRegionIndex(ele_idx);
        if(region_idx==-1)
            throw PhysikaException("Element doesn't belong to any region, can't find its constitutive model in region-wise data.");
        else
            return *constitutive_model_[region_idx];
    }
    else if(material_num == ele_num) //element-wise material
        return *constitutive_model_[ele_idx];
    else
        PHYSIKA_ERROR("Invalid material number.");
}

//explicit instantiations
template class FEMSolidForceModel<float,2>;
template class FEMSolidForceModel<float,3>;
template class FEMSolidForceModel<double,2>;
template class FEMSolidForceModel<double,3>;
    
} //end of namespace Physika
