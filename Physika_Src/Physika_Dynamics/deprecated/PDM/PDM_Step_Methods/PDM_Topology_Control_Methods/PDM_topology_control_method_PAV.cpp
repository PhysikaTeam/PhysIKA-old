/*
 * @file PDM_topology_control_method_PAV.cpp
 * @brief PDMTopologyControlMethodPAV(with Particle At Vertex) used to control the topology of simulated mesh.
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

#include <iostream>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Volumetric_Meshes/tet_mesh.h"
#include "Physika_Dynamics/PDM/PDM_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Topology_Control_Methods/PDM_topology_control_method_PAV.h"

namespace Physika{

template <typename Scalar, int Dim>
void PDMTopologyControlMethodPAV<Scalar, Dim>::addElementTuple(unsigned int fir_ele_id, unsigned int sec_ele_id)
{
    std::cerr<<"error: PDMTopologyControlMethodPAV don't support Fracture!"<<std::endl;
    std::exit(EXIT_FAILURE);
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethodPAV<Scalar, Dim>::topologyControlMethod(Scalar dt)
{
    //udpate mesh vertex position
    if (this->pdm_base_->numSimParticles() != this->mesh_->vertNum())
    {
        std::cerr<<"error: particle size not equal vol_mesh vertex size!"<<std::endl;
        std::exit(EXIT_FAILURE);
    }
    for (unsigned int ver_id = 0; ver_id < this->mesh_->vertNum(); ver_id++)
    {
        const Vector<Scalar, Dim> & ver_pos = this->pdm_base_->particleCurrentPosition(ver_id);
        this->mesh_->setVertPos(ver_id, ver_pos);
    }

}

//explicit instantiations
template class PDMTopologyControlMethodPAV<float, 2>;
template class PDMTopologyControlMethodPAV<float, 3>;
template class PDMTopologyControlMethodPAV<double ,2>;
template class PDMTopologyControlMethodPAV<double, 3>;

}