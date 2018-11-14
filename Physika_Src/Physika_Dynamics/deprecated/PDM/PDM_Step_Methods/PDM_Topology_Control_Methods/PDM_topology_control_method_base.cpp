/*
 * @file PDM_topology_control_method_base.cpp
 * @brief PDMTopologyControlMethod used to control the topology of simulated mesh.
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Geometry/Volumetric_Meshes/tet_mesh.h"
#include "Physika_Dynamics/PDM/PDM_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Topology_Control_Methods/PDM_topology_control_method_base.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMTopologyControlMethodBase<Scalar, Dim>::PDMTopologyControlMethodBase()
    :pdm_base_(NULL), mesh_(NULL)
{

}

template <typename Scalar, int Dim>
PDMTopologyControlMethodBase<Scalar, Dim>::~PDMTopologyControlMethodBase()
{

}

template <typename Scalar, int Dim>
void PDMTopologyControlMethodBase<Scalar, Dim>::setMesh(VolumetricMesh<Scalar, Dim> * mesh)
{
    this->mesh_ = mesh;
    PHYSIKA_ASSERT(this->mesh_);
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethodBase<Scalar, Dim>::setDriver(PDMBase<Scalar, Dim> * pdm_base)
{
    this->pdm_base_ = pdm_base;
    PHYSIKA_ASSERT(this->pdm_base_);
    this->mesh_ = this->pdm_base_->mesh();
    PHYSIKA_ASSERT(this->mesh_);
}

template class PDMTopologyControlMethodBase<float, 2>;
template class PDMTopologyControlMethodBase<float, 3>;
template class PDMTopologyControlMethodBase<double ,2>;
template class PDMTopologyControlMethodBase<double, 3>;

} //end of namespace Physika

