/*
 * @file PDM_topology_control_method_PAV.h 
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

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_TOPOLOGY_CONTROL_METHODS_PDM_TOPOLOGY_CONTROL_METHOD_PAV_H
#define PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_TOPOLOGY_CONTROL_METHODS_PDM_TOPOLOGY_CONTROL_METHOD_PAV_H

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Topology_Control_Methods/PDM_topology_control_method_base.h"

namespace Physika{


template <typename Scalar, int Dim>
class PDMTopologyControlMethodPAV: public PDMTopologyControlMethodBase<Scalar, Dim>
{
public:
    virtual void addElementTuple(unsigned int fir_ele_id, unsigned int sec_ele_id);
    virtual void topologyControlMethod(Scalar dt);
};

} //end of namespace Physika
#endif //PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_TOPOLOGY_CONTROL_METHODS_PDM_TOPOLOGY_CONTROL_METHOD_PAV_H