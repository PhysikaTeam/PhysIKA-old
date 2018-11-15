/*
 * @file PDM_topology_control_method_base.h 
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

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_TOPOLOGY_CONTROL_METHODS_PDM_TOPOLOGY_CONTROL_METHOD_BASE_H
#define PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_TOPOLOGY_CONTROL_METHODS_PDM_TOPOLOGY_CONTROL_METHOD_BASE_H

namespace Physika{

template <typename Scalar, int Dim> class VolumetricMesh;
template<typename Scalar, int Dim> class PDMBase;
class PDMElementTuple;

template <typename Scalar, int Dim>
class PDMTopologyControlMethodBase
{
public:
    PDMTopologyControlMethodBase();
    virtual ~PDMTopologyControlMethodBase();

    virtual void setMesh(VolumetricMesh<Scalar, Dim> * mesh);     
    virtual void setDriver(PDMBase<Scalar, Dim> * pdm_base);

    virtual void addElementTuple(unsigned int fir_ele_id, unsigned int sec_ele_id) = 0;
    virtual void topologyControlMethod(Scalar dt) = 0;

protected:
    VolumetricMesh<Scalar, Dim> * mesh_; //default: NULL
    PDMBase<Scalar, Dim> * pdm_base_;    //default: NULL

};

} //end of namespace Physika
#endif //PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_TOPOLOGY_CONTROL_METHODS_PDM_TOPOLOGY_CONTROL_METHOD_BASE_H
