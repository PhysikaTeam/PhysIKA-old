/*
 * @file invertible_mpm_solid.cpp 
 * @Brief a hybrid of FEM and CPDI2 for large deformation and invertible elasticity, uniform grid.
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

#include "Physika_Geometry/Volumetric_Meshes/quad_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/cubic_mesh.h"
#include "Physika_Dynamics/MPM/invertible_mpm_solid.h"

namespace Physika{

template <typename Scalar, int Dim>
InvertibleMPMSolid<Scalar,Dim>::InvertibleMPMSolid()
{
}

template <typename Scalar, int Dim>
InvertibleMPMSolid<Scalar,Dim>::InvertibleMPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
{
}

template <typename Scalar, int Dim>
InvertibleMPMSolid<Scalar,Dim>::InvertibleMPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file, const Grid<Scalar,Dim> &grid)
{
}

template <typename Scalar, int Dim>
InvertibleMPMSolid<Scalar,Dim>::~InvertibleMPMSolid()
{
}

template <typename Scalar, int Dim>
bool InvertibleMPMSolid<Scalar,Dim>::withRestartSupport() const
{
    return false;
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::write(const std::string &file_name)
{
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::read(const std::string &file_name)
{
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::initSimulationData()
{
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::rasterize()
{
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::solveOnGridForwardEuler(Scalar dt)
{
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::solveOnGridBackwardEuler(Scalar dt)
{
//TO DO
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::resetParticleDomainData()
{
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::constructParticleDomainTopology()
{
}

//explicit instantiation
template class InvertibleMPMSolid<float,2>;
template class InvertibleMPMSolid<double,2>;
template class InvertibleMPMSolid<float,3>;
template class InvertibleMPMSolid<double,3>;

}  //end of namespace Physika
