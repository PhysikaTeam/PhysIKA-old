/*
 * @file fluid_render_task.h 
 * @Basic render task of fluid
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#pragma once

#include <memory>
#include "Physika_Render/Render_Task_Base/render_task_base.h"

namespace Physika{

class FluidRenderUtil;

class FluidRenderTask: public RenderTaskBase
{
public:
    FluidRenderTask(std::shared_ptr<FluidRenderUtil> render_util);
    ~FluidRenderTask();

    //disable copy
    FluidRenderTask(const FluidRenderTask &) = delete;
    FluidRenderTask & operator = (const FluidRenderTask &) = delete;
    
    RenderTaskType type() const override;  //return RenderTaskType::SCREEN_BASED_RENDER_TASK

    void setDrawOpaque(bool draw_opaque);
    void setFluidRadius(float fluid_radius);
    void setFluidBlur(float fluid_blur);
    void setFluidIor(float fluid_ior);
    void setFluidColor(Color<float> fluid_color);

    void setDrawDiffuseParticle(bool draw_diffuse_particle);
    void setDiffuseColor(Color<float> diffuse_color);
    void setDiffuseScale(float diffuse_scale);
    void setDiffuseMotionScale(float diffuse_motion_scale);
    void setDiffuseInscatter(float diffuse_inscatter);
    void setDiffuseOutscatter(float diffuse_outscatter); 

private:
    void renderTaskImpl() override;
    void renderDiffuseParticle(bool front);
    void renderFluidParticle();
    void renderToFullScreenQuad();

    void initFrameBuffers();
    void destroyFrameBuffers();
    void updateFrameBuffers();

    void initShaderPrograms();
    void destroyShaderPrograms();

    void configFakeLightUniforms(bool reverse_light_dir = false); //to delete
    void configCameraUniforms();
    
private:
    std::shared_ptr<FluidRenderUtil> render_util_;

    //------------------------------------------------------------------------------
    float radius_ = 0.01f;
    float fov_ = 3.14159f / 4.0f;

    bool         draw_opaque_ = false;
    float        fluid_blur_ = 1.0f;
    float        fluid_ior_ = 1.0f;
    Color<float> fluid_color_ = { 0.1f, 0.4f, 0.8f, 1.0f };

    bool         draw_diffuse_particle_ = false;
    Color<float> diffuse_color_ = { 1.0f, 1.0f, 1.0f, 1.0f };
    float        diffuse_scale_ = 0.5;
    float        diffuse_motion_scale_ = 1.0f;
    float        diffuse_inscatter_ = 0.8f;
    float        diffuse_outscatter_ = 0.53f;
    bool         diffuse_shadow_ = false;

    //------------------------------------------------------------------------------
    unsigned int cache_screen_width_ = 1280;
    unsigned int cache_screen_height_ = 720;

    unsigned int depth_FBO_ = 0;
    unsigned int depth_TEX_ = 0;
    unsigned int depth_smooth_TEX_ = 0;
    unsigned int depth_zbuffer_RBO_ = 0;

    unsigned int scene_FBO_ = 0;
    unsigned int scene_TEX_ = 0;

    unsigned int thickness_FBO_ = 0;
    unsigned int thickness_TEX_ = 0;
    unsigned int thickness_zbuffer_RBO_;
    //------------------------------------------------------------------------------

    ShaderProgram diffuse_program_;
    ShaderProgram point_thickness_program_;
    ShaderProgram ellipsoid_depth_program_;
    ShaderProgram composite_program_;
    ShaderProgram depth_blur_program_;
};
    
}//end of namespace Physika