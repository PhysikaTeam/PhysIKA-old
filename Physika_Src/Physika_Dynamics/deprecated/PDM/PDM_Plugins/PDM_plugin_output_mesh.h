/*
 * @file PDM_plugin_output_mesh.h 
 * @brief output particle information for mesh generating
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

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_PLUGINS_PDM_PLUGIN_OUTPUT_MESH_H
#define PHYSIKA_DYNAMICS_PDM_PDM_PLUGINS_PDM_PLUGIN_OUTPUT_MESH_H

#include <string>
#include "Physika_Dynamics/PDM/PDM_Plugins/PDM_plugin_base.h"

namespace Physika{

template <typename Scalar> class TetMesh;
template <typename Scalar> class Tri3DMesh;
template <typename Scalar, int Dim> class PDMImpactMethodBase;

template <typename Scalar, int Dim>
class PDMPluginOutputMesh: public PDMPluginBase<Scalar, Dim>
{
public:
    PDMPluginOutputMesh();
    ~PDMPluginOutputMesh();

    //inherited virtual methods
    virtual void onBeginFrame(unsigned int frame);
    virtual void onEndFrame(unsigned int frame);
    virtual void onBeginTimeStep(Scalar time, Scalar dt);
    virtual void onEndTimeStep(Scalar time, Scalar dt);

    void setSaveIntermediateStateTimeStep(int save_intermediate_state_time_step);
    void enableSkipIsolateEle();
    void disableSkipIsolateEle();

    void setImpactMethod(PDMImpactMethodBase<Scalar, Dim> * impact_method);

protected:
    void saveBoundaryMesh(TetMesh<Scalar> * mesh, const std::string & file_name);
    void saveBoundaryMesh(const Tri3DMesh<Scalar> * mesh, const std::string & file_name);
    void saveParticlePos();
    void saveParticleVel();
    void saveVolumetricMesh();
    void saveImpactPos();

protected:
    bool skip_isolate_ele_; //default: false
    int save_intermediate_state_time_step_; //default:-1, used to output particle pos and volumetric mesh

    PDMImpactMethodBase<Scalar, Dim> * impact_method_; //used to output impact_pos

};


}// end of namespace Physika

#endif //PHYSIKA_DYNAMICS_PDM_PDM_PLUGINS_PDM_PLUGIN_OUTPUT_MESH_H