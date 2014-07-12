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
#include "Physika_IO/Volumetric_Mesh_IO/volumetric_mesh_io.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
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
    simulation_mesh_ = VolumetricMesh<Scalar,Dim>::load(file_name);
    if(simulation_mesh_ == NULL)
    {
        std::cerr<<"Failed to load simulation mesh from "<<file_name<<"\n";
        std::exit(EXIT_FAILURE);
    }
}

template <typename Scalar, int Dim>
void FEMBase<Scalar,Dim>::setSimulationMesh(const VolumetricMesh<Scalar,Dim> &mesh)
{
//TO DO
}

}  //end of namespace Physika
