/*
* @file cuda_pbd_sph_plugin_render.h
* @brief base class CudaPBDSPHPluginRender.
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


#ifndef PHYSIKA_DYNAMICS_SPH_GPU_SPH_CUDA_PBD_SPH_PLUGIN_RENDER_H_
#define PHYSIKA_DYNAMICS_SPH_GPU_SPH_CUDA_PBD_SPH_PLUGIN_RENDER_H_

#include "vector_types.h"
#include "driver_types.h"
#include "Physika_Dynamics/SPH/sph_plugin_base.h"

namespace Physika{

class GlutWindow;
class ScreenBasedRenderManager;
//class FluidRender;


class CudaPBDSPHPluginRender : public SPHPluginBase<float, 3>
{
public:
    CudaPBDSPHPluginRender();
    ~CudaPBDSPHPluginRender();


    //inherited virtual methods
    virtual void onBeginFrame(unsigned int frame);
    virtual void onEndFrame(unsigned int frame);
    virtual void onBeginTimeStep(float time, float dt);
    virtual void onEndTimeStep(float time, float dt);

    //setter & getter
    GlutWindow * window();
    void setWindow(GlutWindow * window);

private:

    float4 * mapVBOBuffer();
    void unmapVBOBuffer();

    static void initFunction();
    static void displayFunction();
    
    void activateCurrentInstance();

private:

    GlutWindow * window_;

    ScreenBasedRenderManager * screen_based_render_manager_ = NULL;
    //FluidRender * fluid_render_ = NULL;

    cudaGraphicsResource * cuda_pos_vbo_resource_;

    static CudaPBDSPHPluginRender * active_instance_;
};



}//end of namespace Physika

#endif //PHYSIKA_DYNAMICS_SPH_GPU_SPH_CUDA_PBD_SPH_PLUGIN_RENDER_H_