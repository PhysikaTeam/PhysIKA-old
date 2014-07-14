/*
 * @file fem_isotropic_hyperelastic_solid.cpp 
 * @Brief FEM driver for isotropic hyperelastic solids, not necessarilly homogeneous.
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

#include "Physika_Dynamics/Constitutive_Models/isotropic_hyperelastic_material.h"
#include "Physika_Dynamics/Constitutive_Models/isotropic_linear_elasticity.h"
#include "Physika_Dynamics/Constitutive_Models/neo_hookean.h"
#include "Physika_Dynamics/Constitutive_Models/st_venant_kirchhoff.h"
#include "Physika_Dynamics/FEM/fem_isotropic_hyperelastic_solid.h"

namespace Physika{

template <typename Scalar, int Dim>
FEMIsotropicHyperelasticSolid<Scalar,Dim>::FEMIsotropicHyperelasticSolid()
    :FEMBase<Scalar,Dim>()
{
}

template <typename Scalar, int Dim>
FEMIsotropicHyperelasticSolid<Scalar,Dim>::FEMIsotropicHyperelasticSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :FEMBase<Scalar,Dim>(start_frame,end_frame,frame_rate,max_dt,write_to_file)
{
}

template <typename Scalar, int Dim>
FEMIsotropicHyperelasticSolid<Scalar,Dim>::~FEMIsotropicHyperelasticSolid()
{
    for(unsigned int i = 0; i < constitutive_model_.size(); ++i)
        delete constitutive_model_[i];
}

template <typename Scalar, int Dim>
void FEMIsotropicHyperelasticSolid<Scalar,Dim>::initialize()
{//TO DO
}

template <typename Scalar, int Dim>
void FEMIsotropicHyperelasticSolid<Scalar,Dim>::advanceStep(Scalar dt)
{//TO DO
}

template <typename Scalar, int Dim>
void FEMIsotropicHyperelasticSolid<Scalar,Dim>::write(const std::string &file_name)
{//TO DO
}

template <typename Scalar, int Dim>
void FEMIsotropicHyperelasticSolid<Scalar,Dim>::read(const std::string &file_name)
{//TO DO
}

template <typename Scalar, int Dim>
void FEMIsotropicHyperelasticSolid<Scalar,Dim>::addPlugin(DriverPluginBase<Scalar> *plugin)
{//TO DO
}

template <typename Scalar, int Dim>
void FEMIsotropicHyperelasticSolid<Scalar,Dim>::initConfiguration(const std::string &file_name)
{//TO DO
}

template <typename Scalar, int Dim>
unsigned int FEMIsotropicHyperelasticSolid<Scalar,Dim>::materialNum() const
{
    return constitutive_model_.size();
}

template <typename Scalar, int Dim>
void FEMIsotropicHyperelasticSolid<Scalar,Dim>::setHomogeneousMaterial(const IsotropicHyperelasticMaterial<Scalar,Dim> &material)
{
    constitutive_model_.clear();
//TO DO
}

template <typename Scalar, int Dim>
void FEMIsotropicHyperelasticSolid<Scalar,Dim>::setRegionWiseMaterial(const std::vector<IsotropicHyperelasticMaterial<Scalar,Dim>*> &materials)
{
//TO DO
}

template <typename Scalar, int Dim>
void FEMIsotropicHyperelasticSolid<Scalar,Dim>::setElementWiseMaterial(const std::vector<IsotropicHyperelasticMaterial<Scalar,Dim>*> &materials)
{
//TO DO
}

//explicit instantiations
template class FEMIsotropicHyperelasticSolid<float,2>;
template class FEMIsotropicHyperelasticSolid<double,2>;
template class FEMIsotropicHyperelasticSolid<float,3>;
template class FEMIsotropicHyperelasticSolid<double,3>;

}  //end of namespace Physika
