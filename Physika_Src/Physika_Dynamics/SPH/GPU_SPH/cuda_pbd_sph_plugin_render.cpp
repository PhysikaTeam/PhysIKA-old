/*
* @file cuda_pbd_sph_plugin_render.cpp
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

#include <vector>
#include <GL/freeglut.h>
#include "cuda_gl_interop.h" 

#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Vectors/vector_4d.h"
#include "Physika_Core/Utilities/physika_exception.h"

#include "Physika_GUI/Glut_Window/glut_window.h"
#include "Physika_Render/Screen_Based_Render_Manager/screen_based_render_manager.h"
#include "Physika_Render/Screen_Based_Render_Manager/Fluid_Render/fluid_render.h"

#include "Physika_Dynamics/SPH/GPU_SPH/cuda_pbd_sph.h"
#include "Physika_Dynamics/SPH/GPU_SPH/cuda_pbd_sph_plugin_render.h"

namespace Physika{

CudaPBDSPHPluginRender * CudaPBDSPHPluginRender::active_instance_ = NULL;

CudaPBDSPHPluginRender::CudaPBDSPHPluginRender()
{
    this->activateCurrentInstance();
}

CudaPBDSPHPluginRender::~CudaPBDSPHPluginRender()
{
    //need further consideration
    delete this->screen_based_render_manager_;
    delete this->fluid_render_;
}

void CudaPBDSPHPluginRender::onBeginFrame(unsigned int frame)
{
    // to do
}

void CudaPBDSPHPluginRender::onEndFrame(unsigned int frame)
{
    // to do
}

void CudaPBDSPHPluginRender::onBeginTimeStep(float time, float dt)
{
    // to do
}


GlutWindow * CudaPBDSPHPluginRender::window()
{
    return this->window_;
}

void CudaPBDSPHPluginRender::setWindow(GlutWindow * window)
{
    if (window == NULL)
        throw PhysikaException("error: NULL window pointer provided to render plugin! \n");
    
    this->window_ = window;
    this->window_->setInitFunction(CudaPBDSPHPluginRender::initFunction);
    this->window_->setDisplayFunction(CudaPBDSPHPluginRender::displayFunction);
}

float4 * CudaPBDSPHPluginRender::mapVBOBuffer()
{
    float4 * vbo_device_ptr = NULL;
    size_t size = 0;

    cudaGraphicsMapResources(1, &this->cuda_pos_vbo_resource_, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&vbo_device_ptr, &size, this->cuda_pos_vbo_resource_);

    return vbo_device_ptr;
}

void CudaPBDSPHPluginRender::unmapVBOBuffer()
{
    cudaGraphicsUnmapResources(1, &this->cuda_pos_vbo_resource_, 0);
}

void CudaPBDSPHPluginRender::initFunction()
{
    active_instance_->screen_based_render_manager_ = new ScreenBasedRenderManager(active_instance_->window_);

    //need further consideration
    active_instance_->screen_based_render_manager_->addPlane({ 0.0f, 1.0f, 0.0f, 0.0f });

    active_instance_->screen_based_render_manager_->setLightPos({ 4.35653f, 12.5529f, 5.77261f });
    active_instance_->screen_based_render_manager_->setLightTarget({ 1.508125f, 1.00766f, 0.0f });
    active_instance_->screen_based_render_manager_->setLightFov(28.0725f);
    active_instance_->screen_based_render_manager_->setLightSpotMin(0.0);
    active_instance_->screen_based_render_manager_->setLightSpotMax(0.5);

    ///////////////////////////////////////////////////////////////////////////////////////////////

    unsigned int screen_width = active_instance_->window_->width();
    unsigned int screen_height = active_instance_->window_->height();

    CudaPBDSPH * cuda_pbd_sph = dynamic_cast<CudaPBDSPH *>(active_instance_->driver());
    unsigned int fluid_particle_num = cuda_pbd_sph->numFluidParticle();

    active_instance_->fluid_render_ = new FluidRender(fluid_particle_num, 0, screen_width, screen_height);

    std::vector<unsigned int> indices_vec;
    for (unsigned int id = 0; id < fluid_particle_num; ++id)
        indices_vec.push_back(id);

    std::vector<float> anisotropy_vec;
    for(unsigned int id = 0; id < fluid_particle_num; ++id)
    {
        anisotropy_vec.push_back(0.5);
        anisotropy_vec.push_back(0.5);
        anisotropy_vec.push_back(0.5);
        anisotropy_vec.push_back(0.5);
    }

    active_instance_->fluid_render_->updateFluidParticleBuffer(NULL, NULL, anisotropy_vec.data(), anisotropy_vec.data(), anisotropy_vec.data(), indices_vec.data(), fluid_particle_num);


    active_instance_->fluid_render_->setDrawFluidParticle(false);
    active_instance_->fluid_render_->setDrawPoint(true);
    active_instance_->fluid_render_->setFluidRadius(0.0125f);

    active_instance_->screen_based_render_manager_->setFluidRender(active_instance_->fluid_render_);

    ///////////////////////////////////////////////////////////////////////////////////////////////

    GLuint pos_VBO = active_instance_->fluid_render_->fluidPositionVBO();
    cudaGraphicsGLRegisterBuffer(&active_instance_->cuda_pos_vbo_resource_, pos_VBO, cudaGraphicsMapFlagsWriteDiscard);


}

void CudaPBDSPHPluginRender::displayFunction()
{
    active_instance_->screen_based_render_manager_->render();
}

void CudaPBDSPHPluginRender::activateCurrentInstance()
{
    this->active_instance_ = this;
}


}//end of namespace Physika