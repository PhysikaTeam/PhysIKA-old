/*
 * @file quad_cubic_mesh_fem_solid_force_model.cpp 
 * @Brief fem solid force model for quad and cubic mesh.
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

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Dynamics/FEM/fem_solid.h"
#include "Physika_Dynamics/FEM/FEM_Solid_Force_Model/quad_cubic_mesh_fem_solid_force_model.h"

namespace Physika{

template <typename Scalar, int Dim>
QuadCubicMeshFEMSolidForceModel<Scalar,Dim>::QuadCubicMeshFEMSolidForceModel()
    :FEMSolidForceModel<Scalar,Dim>()
{
}

template <typename Scalar, int Dim>
QuadCubicMeshFEMSolidForceModel<Scalar,Dim>::QuadCubicMeshFEMSolidForceModel(const FEMSolid<Scalar,Dim> *fem_solid_driver)
    :FEMSolidForceModel<Scalar,Dim>(fem_solid_driver)
{
}

template <typename Scalar, int Dim>
QuadCubicMeshFEMSolidForceModel<Scalar,Dim>::~QuadCubicMeshFEMSolidForceModel()
{
}

template <typename Scalar, int Dim>
void QuadCubicMeshFEMSolidForceModel<Scalar,Dim>::computeGlobalInternalForces(const std::vector<Vector<Scalar,Dim> > &current_vert_pos, std::vector<Vector<Scalar,Dim> > &force) const
{
}

template <typename Scalar, int Dim>
void QuadCubicMeshFEMSolidForceModel<Scalar,Dim>::computeElementInternalForces(unsigned int ele_idx, const std::vector<Vector<Scalar,Dim> > &current_vert_pos, std::vector<Vector<Scalar,Dim> > &force) const
{
}

template <typename Scalar, int Dim>
void QuadCubicMeshFEMSolidForceModel<Scalar,Dim>::computeGlobalInternalForceDifferentials(const std::vector<Vector<Scalar,Dim> > &current_vert_pos, const std::vector<Vector<Scalar,Dim> > &vert_pos_differentials, std::vector<Vector<Scalar,Dim> > &force_differentials) const
{
}

template <typename Scalar, int Dim>
void QuadCubicMeshFEMSolidForceModel<Scalar,Dim>::computeElementInternalForceDifferentials(unsigned int ele_idx, const std::vector<Vector<Scalar,Dim> > &current_vert_pos, const std::vector<Vector<Scalar,Dim> > &vert_pos_differentials, std::vector<Vector<Scalar,Dim> > &force_differentials) const
{
}

//explicit instantiations
template class QuadCubicMeshFEMSolidForceModel<float,2>;
template class QuadCubicMeshFEMSolidForceModel<float,3>;
template class QuadCubicMeshFEMSolidForceModel<double,2>;
template class QuadCubicMeshFEMSolidForceModel<double,3>;

} //end of namespace Physika
