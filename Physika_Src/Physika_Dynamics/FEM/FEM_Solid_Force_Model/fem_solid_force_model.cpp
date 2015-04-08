/*
 * @file fem_solid_force_model.cpp 
 * @Brief the "engine" for fem solid drivers.
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

#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Dynamics/FEM/fem_solid.h"
#include "Physika_Dynamics/FEM/FEM_Solid_Force_Model/fem_solid_force_model.h"

namespace Physika{

template <typename Scalar, int Dim>
FEMSolidForceModel<Scalar,Dim>::FEMSolidForceModel()
    :fem_solid_driver_(NULL)
{
}

template <typename Scalar, int Dim>
FEMSolidForceModel<Scalar,Dim>::FEMSolidForceModel(const FEMSolid<Scalar,Dim> *fem_solid_driver)
    :fem_solid_driver_(fem_solid_driver)
{
}

template <typename Scalar, int Dim>    
FEMSolidForceModel<Scalar,Dim>::~FEMSolidForceModel()
{
    if(fem_solid_driver_)
        fem_solid_driver_ = NULL;
}

template <typename Scalar, int Dim>    
const FEMSolid<Scalar,Dim>* FEMSolidForceModel<Scalar,Dim>::driver() const
{
    return fem_solid_driver_;
}

template <typename Scalar, int Dim>
void FEMSolidForceModel<Scalar,Dim>::setDriver(const FEMSolid<Scalar,Dim> *fem_solid_driver)
{
    if(fem_solid_driver == NULL)
        throw PhysikaException("NULL FEMSolidDriver passed to FEMSolidForceModel!");
    fem_solid_driver_ = fem_solid_driver;
}

//explicit instantiations
template class FEMSolidForceModel<float,2>;
template class FEMSolidForceModel<float,3>;
template class FEMSolidForceModel<double,2>;
template class FEMSolidForceModel<double,3>;
    
} //end of namespace Physika
