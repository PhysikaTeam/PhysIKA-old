/*
 * @file fluid_render_task.cpp
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


#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"
#include "Physika_Render/Render_Scene_Config/render_scene_config.h"

#include "fluid_render_task.h"
#include "fluid_render_util.h"
#include "fluid_render_shader_srcs.h"

namespace Physika {


FluidRenderTask::FluidRenderTask(std::shared_ptr<FluidRenderUtil> render_util)
    :render_util_(std::move(render_util))
{
    //Note: we do not init shader_ member of RenderTaskBase
    this->disableBindShader();

    this->initFrameBuffers();
    this->initShaderPrograms();
}

FluidRenderTask::~FluidRenderTask()
{
    this->destroyFrameBuffers();
    this->destroyShaderPrograms();
}

RenderTaskType FluidRenderTask::type() const
{
    return RenderTaskType::SCREEN_BASED_RENDER_TASK;
}

void FluidRenderTask::setDrawOpaque(bool draw_opaque)
{
    this->draw_opaque_ = draw_opaque;
}

void FluidRenderTask::setFluidRadius(float fluid_radius)
{
    this->radius_ = fluid_radius;
}

void FluidRenderTask::setFluidBlur(float fluid_blur)
{
    this->fluid_blur_ = fluid_blur;
}

void FluidRenderTask::setFluidIor(float fluid_ior)
{
    this->fluid_ior_ = fluid_ior;
}

void FluidRenderTask::setFluidColor(Color<float> fluid_color)
{
    this->fluid_color_ = fluid_color;
}

void FluidRenderTask::setDrawDiffuseParticle(bool draw_diffuse_particle)
{
    this->draw_diffuse_particle_ = draw_diffuse_particle;
}

void FluidRenderTask::setDiffuseColor(Color<float> diffuse_color)
{
    this->diffuse_color_ = diffuse_color;
}

void FluidRenderTask::setDiffuseScale(float diffuse_scale)
{
    this->diffuse_scale_ = diffuse_scale;
}

void FluidRenderTask::setDiffuseMotionScale(float diffuse_motion_scale)
{
    this->diffuse_motion_scale_ = diffuse_motion_scale;
}

void FluidRenderTask::setDiffuseInscatter(float diffuse_inscatter)
{
    this->diffuse_inscatter_ = diffuse_inscatter;
}

void FluidRenderTask::setDiffuseOutscatter(float diffuse_outscatter)
{
    this->diffuse_outscatter_ = diffuse_outscatter;
}

void FluidRenderTask::initFrameBuffers()
{
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
    cache_screen_width_ = render_scene_config.screenWidth();
    cache_screen_height_ = render_scene_config.screenHeight();

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //create depth_TEX
    glVerify(glGenTextures(1, &this->depth_TEX_));
    glVerify(glBindTexture(GL_TEXTURE_RECTANGLE, this->depth_TEX_));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    glVerify(glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_LUMINANCE32F_ARB, cache_screen_width_, cache_screen_height_, 0, GL_LUMINANCE, GL_FLOAT, NULL));
    glVerify(glBindTexture(GL_TEXTURE_RECTANGLE, 0));

    //create zbuffer_RBO
    glVerify(glGenRenderbuffers(1, &depth_zbuffer_RBO_));
    glVerify(glBindRenderbuffer(GL_RENDERBUFFER, depth_zbuffer_RBO_));
    glVerify(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, cache_screen_width_, cache_screen_height_));
    glVerify(glBindRenderbuffer(GL_RENDERBUFFER, 0));

    //create depth_FBO, bind depth_TEX & zbuffer_RBO to depth_FBO
    glVerify(glGenFramebuffers(1, &this->depth_FBO_));
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, this->depth_FBO_));
    glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_ARB, this->depth_TEX_, 0));
    glVerify(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_zbuffer_RBO_));

    glVerify(glCheckFramebufferStatus(GL_FRAMEBUFFER));

    //specify which buffer to draw and read
    glVerify(glDrawBuffer(GL_COLOR_ATTACHMENT0));
    glVerify(glReadBuffer(GL_COLOR_ATTACHMENT0));

    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, 0));

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //create depth_smooth_TEX
    glVerify(glGenTextures(1, &this->depth_smooth_TEX_));
    glVerify(glBindTexture(GL_TEXTURE_2D, this->depth_smooth_TEX_));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, cache_screen_width_, cache_screen_height_, 0, GL_LUMINANCE, GL_FLOAT, NULL));
    glVerify(glBindTexture(GL_TEXTURE_2D, 0));

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //create scene_TEX
    glVerify(glGenTextures(1, &this->scene_TEX_));

    glVerify(glBindTexture(GL_TEXTURE_2D, this->scene_TEX_));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cache_screen_width_, cache_screen_height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));
    glVerify(glBindTexture(GL_TEXTURE_2D, 0));

    //create scene_FBO & bind secene_TEX to scene_FBO
    glVerify(glGenFramebuffers(1, &this->scene_FBO_));
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, this->scene_FBO_));
    glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->scene_TEX_, 0));
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, 0));

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //create thickness_TEX
    glVerify(glGenTextures(1, &this->thickness_TEX_));
    glVerify(glBindTexture(GL_TEXTURE_2D, this->thickness_TEX_));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cache_screen_width_, cache_screen_height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));
    glVerify(glBindTexture(GL_TEXTURE_2D, 0));

    //create thick_zuffer_RBO and bind it to thickness_FBO
    glVerify(glGenRenderbuffers(1, &thickness_zbuffer_RBO_));
    glVerify(glBindRenderbuffer(GL_RENDERBUFFER, thickness_zbuffer_RBO_));
    glVerify(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, cache_screen_width_, cache_screen_height_));
    glVerify(glBindRenderbuffer(GL_RENDERBUFFER, 0));

    //create thickness_FBO and bind thickness_TEX & thick_zuffer_RBO to it
    glVerify(glGenFramebuffers(1, &this->thickness_FBO_));
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, this->thickness_FBO_));
    glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->thickness_TEX_, 0));
    glVerify(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, thickness_zbuffer_RBO_));
    glVerify(glCheckFramebufferStatus(GL_FRAMEBUFFER));
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

void FluidRenderTask::destroyFrameBuffers()
{
    glVerify(glDeleteFramebuffers(1, &this->depth_FBO_));
    glVerify(glDeleteTextures(1, &this->depth_TEX_));
    glVerify(glDeleteTextures(1, &this->depth_smooth_TEX_));
    glVerify(glDeleteRenderbuffers(1, &this->depth_zbuffer_RBO_));

    glVerify(glDeleteFramebuffers(1, &this->scene_FBO_));
    glVerify(glDeleteTextures(1, &this->scene_TEX_));

    glVerify(glDeleteFramebuffers(1, &this->thickness_FBO_));
    glVerify(glDeleteTextures(1, &this->thickness_TEX_));
    glVerify(glDeleteRenderbuffers(1, &this->thickness_zbuffer_RBO_));
}

void FluidRenderTask::updateFrameBuffers()
{
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
    unsigned int screen_width = render_scene_config.screenWidth();
    unsigned int screen_height = render_scene_config.screenHeight();

    if (cache_screen_width_ != screen_width || cache_screen_height_ != screen_height)
    {
        this->destroyFrameBuffers();
        this->initFrameBuffers();
    }
}

void FluidRenderTask::initShaderPrograms()
{
    //diffuse_program
    this->diffuse_program_.createFromCStyleString(vertex_diffuse_shader, fragment_diffuse_shader, geometry_diffuse_shader);
    glVerify(glProgramParameteri(this->diffuse_program_.id(), GL_GEOMETRY_VERTICES_OUT_EXT, 4));
    glVerify(glProgramParameteri(this->diffuse_program_.id(), GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS));
    glVerify(glProgramParameteri(this->diffuse_program_.id(), GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP));
    glVerify(glLinkProgram(this->diffuse_program_.id()));

    //point_thickness_program
    this->point_thickness_program_.createFromCStyleString(vertex_point_depth_shader, fragment_point_thickness_shader);

    //ellipsoid_thickness_program
    this->ellipsoid_depth_program_.createFromCStyleString(vertex_ellipsoid_depth_shader, fragment_ellipsoid_depth_shader, geometry_ellipsoid_depth_shader);
    glVerify(glProgramParameteri(this->ellipsoid_depth_program_.id(), GL_GEOMETRY_VERTICES_OUT_EXT, 4));
    glVerify(glProgramParameteri(this->ellipsoid_depth_program_.id(), GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS));
    glVerify(glProgramParameteri(this->ellipsoid_depth_program_.id(), GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP));
    glVerify(glLinkProgram(this->ellipsoid_depth_program_.id()));

    //composite_program
    this->composite_program_.createFromCStyleString(vertex_pass_through_shader, fragment_composite_shader);

    //depth_blur_program
    this->depth_blur_program_.createFromCStyleString(vertex_pass_through_shader, fragment_blur_depth_shader);
}

void FluidRenderTask::destroyShaderPrograms()
{
    this->diffuse_program_.destory();
    this->point_thickness_program_.destory();
    this->ellipsoid_depth_program_.destory();
    this->composite_program_.destory();
    this->depth_blur_program_.destory();
}

void FluidRenderTask::renderTaskImpl()
{
    //resize frame buffers if window size change
    this->updateFrameBuffers();

    if (this->draw_diffuse_particle_)
        this->renderDiffuseParticle(false);

    // render fluid surface
    this->renderFluidParticle();

    // second pass of diffuse particles for particles in front of fluid surface
    if (this->draw_diffuse_particle_)
        this->renderDiffuseParticle(true);

}

void FluidRenderTask::renderDiffuseParticle(bool front)
{

    //warning: this function has not been tested yet!


    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();

    unsigned int screen_width = render_scene_config.screenWidth();
    unsigned int screen_height = render_scene_config.screenHeight();
    float        screen_aspect = static_cast<float>(screen_width) / screen_height;


    glEnable(GL_POINT_SPRITE);
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    glDepthMask(GL_FALSE);
    glEnable(GL_DEPTH_TEST);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    glDisable(GL_CULL_FACE);

    this->diffuse_program_.use();
    this->configCameraUniforms();
    this->configFakeLightUniforms();

    float thickness_scale = 1;
    openGLSetCurBindShaderFloat("pointRadius", screen_width / thickness_scale / (2.0f*screen_aspect*tanf(this->fov_*0.5f)));

    glm::vec2 inv_viewport = { 1.0f / screen_width, 1.0f / screen_height };
    openGLSetCurBindShaderVec2("invViewport", inv_viewport);

    openGLSetCurBindShaderCol4("color", this->diffuse_color_);

    openGLSetCurBindShaderFloat("motionBlurScale", this->diffuse_motion_scale_);
    openGLSetCurBindShaderFloat("diffusion", 1.0f);
    openGLSetCurBindShaderFloat("pointScale", this->radius_*this->diffuse_scale_);
    openGLSetCurBindShaderFloat("inscatterCoefficient", this->diffuse_inscatter_);
    openGLSetCurBindShaderFloat("outscatterCoefficient", this->diffuse_outscatter_);
    openGLSetCurBindShaderInt("tex", 0); //to modify?
    
    float shadow_taps[24] = {
                              -0.326212f, -0.40581f,  -0.840144f,  -0.07358f,
                              -0.695914f,  0.457137f, -0.203345f,   0.620716f,
                               0.96234f,  -0.194983f,  0.473434f,  -0.480026f,
                               0.519456f,  0.767022f,  0.185461f,  -0.893124f,
                               0.507431f,  0.064425f,  0.89642f,    0.412458f,
                              -0.32194f,  -0.932615f,  -0.791559f, -0.59771f
                            };

    openGLSetCurBindShaderFloat2V("shadowTaps", 12, shadow_taps);

    openGLSetCurBindShaderInt("depthTex", 0);
    openGLSetCurBindShaderInt("shadowTex", 1);
    openGLSetCurBindShaderInt("noiseTex", 2);


    openGLSetCurBindShaderInt("front", front);
    openGLSetCurBindShaderInt("shadow", this->diffuse_shadow_);

    //depth smooth tex
    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, this->depth_smooth_TEX_);

    //shadow tex
    glActiveTexture(GL_TEXTURE1);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE));


    glClientActiveTexture(GL_TEXTURE1);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glVerify(glBindBuffer(GL_ARRAY_BUFFER, render_util_->diffuseParticleVelocityVBO()));
    glTexCoordPointer(4, GL_FLOAT, sizeof(float) * 4, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, render_util_->diffuseParticlePositionVBO());
    glVertexPointer(4, GL_FLOAT, sizeof(float) * 4, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, render_util_->diffuseParticleEBO());

    //need further consideration
    glDrawElements(GL_POINTS, render_util_->diffuseParticleNum(), GL_UNSIGNED_INT, 0);

    this->diffuse_program_.unUse();

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);

    glDisable(GL_POINT_SPRITE);
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);

    glVerify(glActiveTexture(GL_TEXTURE0));
    glVerify(glDisable(GL_TEXTURE_2D));
    glVerify(glActiveTexture(GL_TEXTURE1));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE));
    glVerify(glDisable(GL_TEXTURE_2D));
    glVerify(glActiveTexture(GL_TEXTURE2));
    glVerify(glDisable(GL_TEXTURE_2D));
}

void FluidRenderTask::renderFluidParticle()
{
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
    ScreenBasedRenderManager & render_manager = render_scene_config.screenBasedRenderManager();

    unsigned int screen_width = render_scene_config.screenWidth();
    unsigned int screen_height = render_scene_config.screenHeight();
    float        screen_aspect = static_cast<float>(screen_width) / screen_height;

    /********************************************************************************************************************/

    GLuint msaa_FBO = render_manager.msaaFBO();

    // resolve msaa back buffer to texture
    glVerify(glBindFramebuffer(GL_READ_FRAMEBUFFER, msaa_FBO));
    glVerify(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, this->scene_FBO_));
    glVerify(glBlitFramebuffer(0, 0, screen_width, screen_height, 0, 0, screen_width, screen_height, GL_COLOR_BUFFER_BIT, GL_LINEAR));
    
    /********************************************************************************************************************/
   
    //switch to thickness_FBO
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, this->thickness_FBO_));
    glViewport(0, 0, screen_width, screen_height);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_DEPTH_BUFFER_BIT);

    glDepthMask(GL_TRUE);
    glDisable(GL_CULL_FACE);

    //draw shapes
    render_manager.renderAllNormalRenderTasks();

    /********************************************************************************************************************/

    glClear(GL_COLOR_BUFFER_BIT);
    

    glEnable(GL_POINT_SPRITE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

  
    this->point_thickness_program_.use();
    this->configCameraUniforms();

    // make sprites larger to get smoother thickness texture
    const float thickness_scale = 4.0f;
    openGLSetCurBindShaderFloat("pointRadius", thickness_scale*this->radius_);
    openGLSetCurBindShaderFloat("pointScale", screen_width / screen_aspect * (1.0f / (tanf(this->fov_*0.5f))));

    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, render_util_->fluidParticlePositionVBO());
    glVertexPointer(3, GL_FLOAT, sizeof(float) * 4, nullptr);

    //---------------------------------------------------------------------------
    //need further consideration about fluid_particle_num
    glDrawArrays(GL_POINTS, 0, render_util_->fluidParticleNum());
    //---------------------------------------------------------------------------

    this->point_thickness_program_.unUse();

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_POINT_SPRITE);
    glDisable(GL_BLEND);

    /********************************************************************************************************************/

    //switch to depth_FBO
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, this->depth_FBO_));
    glVerify(glViewport(0, 0, screen_width, screen_height));

    glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_ARB, this->depth_TEX_, 0));
    glVerify(glDrawBuffer(GL_COLOR_ATTACHMENT0));

    glDisable(GL_POINT_SPRITE);
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

    
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    this->ellipsoid_depth_program_.use();
    this->configCameraUniforms();

    glm::vec3    inv_viewport = { 1.0f / screen_width, 1.0f / screen_height, 1.0f };
    openGLSetCurBindShaderVec3("invViewport", inv_viewport);

    const float view_height = tanf(this->fov_ / 2.0f);
    glm::vec3 inv_projection = { screen_aspect*view_height, view_height, 1.0f };
    openGLSetCurBindShaderVec3("invProjection", inv_projection);

    glVerify(glEnableClientState(GL_VERTEX_ARRAY));
    glVerify(glBindBuffer(GL_ARRAY_BUFFER, render_util_->fluidParticlePositionVBO()));
    glVerify(glVertexPointer(3, GL_FLOAT, sizeof(float) * 4, nullptr));

    // ellipsoid eigenvectors
    int s1 = glGetAttribLocation(this->ellipsoid_depth_program_.id(), "q1");
    glVerify(glEnableVertexAttribArray(s1));
    glVerify(glBindBuffer(GL_ARRAY_BUFFER, render_util_->fluidParticleAnisotropyVBO(0)));
    glVerify(glVertexAttribPointer(s1, 4, GL_FLOAT, GL_FALSE, 0, nullptr));

    int s2 = glGetAttribLocation(this->ellipsoid_depth_program_.id(), "q2");
    glVerify(glEnableVertexAttribArray(s2));
    glVerify(glBindBuffer(GL_ARRAY_BUFFER, render_util_->fluidParticleAnisotropyVBO(1)));
    glVerify(glVertexAttribPointer(s2, 4, GL_FLOAT, GL_FALSE, 0, nullptr));

    int s3 = glGetAttribLocation(this->ellipsoid_depth_program_.id(), "q3");
    glVerify(glEnableVertexAttribArray(s3));
    glVerify(glBindBuffer(GL_ARRAY_BUFFER, render_util_->fluidParticleAnisotropyVBO(2)));
    glVerify(glVertexAttribPointer(s3, 4, GL_FLOAT, GL_FALSE, 0, nullptr));
    

    //---------------------------------------------------------------------------
    //need further consideration about fluid_particle_num
    glVerify(glDrawArrays(GL_POINTS, 0, render_util_->fluidParticleNum()));
    //---------------------------------------------------------------------------
    
    this->ellipsoid_depth_program_.unUse();

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableVertexAttribArray(s1);
    glDisableVertexAttribArray(s2);
    glDisableVertexAttribArray(s3);

    glDisable(GL_POINT_SPRITE);

    /********************************************************************************************************************/
    // blur
    glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->depth_smooth_TEX_, 0));
    
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_RECTANGLE);
    glBindTexture(GL_TEXTURE_RECTANGLE, this->depth_TEX_);

    glActiveTexture(GL_TEXTURE1);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, this->thickness_TEX_);

    this->depth_blur_program_.use();

    openGLSetCurBindShaderFloat("blurRadiusWorld", this->radius_*0.5f);	// blur half the radius by default
    openGLSetCurBindShaderFloat("blurScale", screen_width / screen_aspect * (1.0f / (tanf(this->fov_*0.5f))));

    glm::vec2 inv_text_scale = { 1.0f / screen_aspect, 1.0f };
    openGLSetCurBindShaderVec2("invTexScale", inv_text_scale);
    openGLSetCurBindShaderFloat("blurFalloff", this->fluid_blur_);
    openGLSetCurBindShaderInt("depthTex", 0);
    openGLSetCurBindShaderInt("thicknessTex", 1);
    openGLSetCurBindShaderInt("debug", this->draw_opaque_);

    //---------------------------------------------------------------------------
    //render to depth_FBO
    glVerify(this->renderToFullScreenQuad());
    //---------------------------------------------------------------------------

    glActiveTexture(GL_TEXTURE0);
    glDisable(GL_TEXTURE_RECTANGLE);

    /********************************************************************************************************************/

    //switch to msaa_FBO, composite with scene
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, msaa_FBO));

    glVerify(glEnable(GL_DEPTH_TEST));
    glVerify(glDepthMask(GL_TRUE));
    glVerify(glDisable(GL_BLEND));
    glVerify(glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA));

    // smoothed depth tex
    glVerify(glActiveTexture(GL_TEXTURE0));
    glVerify(glEnable(GL_TEXTURE_2D));
    glVerify(glBindTexture(GL_TEXTURE_2D, this->depth_smooth_TEX_));

    // shadow tex
    glVerify(glActiveTexture(GL_TEXTURE1));
    glVerify(glEnable(GL_TEXTURE_2D));
    glVerify(glBindTexture(GL_TEXTURE_2D, 0));

    // thickness tex
    glVerify(glActiveTexture(GL_TEXTURE2));
    glVerify(glEnable(GL_TEXTURE_2D));
    glVerify(glBindTexture(GL_TEXTURE_2D, this->thickness_TEX_));

    // scene tex
    glVerify(glActiveTexture(GL_TEXTURE3));
    glVerify(glEnable(GL_TEXTURE_2D));
    glVerify(glBindTexture(GL_TEXTURE_2D, this->scene_TEX_));

    this->composite_program_.use();
    this->configFakeLightUniforms();

    openGLSetCurBindShaderVec2("invTexScale", glm::vec2(1.0f / screen_width, 1.0f / screen_height));
    openGLSetCurBindShaderVec2("clipPosToEye", glm::vec2(tanf(this->fov_*0.5f)*screen_aspect, tanf(this->fov_*0.5f)));
    openGLSetCurBindShaderCol4("color", this->fluid_color_);
    openGLSetCurBindShaderFloat("ior", this->fluid_ior_);
    openGLSetCurBindShaderInt("debug", this->draw_opaque_);

    openGLSetCurBindShaderInt("tex", 0);
    openGLSetCurBindShaderInt("shadowTex", 1);
    openGLSetCurBindShaderInt("thicknessTex", 2);
    openGLSetCurBindShaderInt("sceneTex", 3);
    openGLSetCurBindShaderInt("reflectTex", 5);

    float shadow_taps[24] = {
                                -0.326212f, -0.40581f,  -0.840144f,  -0.07358f,
                                -0.695914f,  0.457137f, -0.203345f,   0.620716f,
                                 0.96234f,  -0.194983f,  0.473434f,  -0.480026f,
                                 0.519456f,  0.767022f,  0.185461f,  -0.893124f,
                                 0.507431f,  0.064425f,  0.89642f,    0.412458f,
                                -0.32194f,  -0.932615f,  -0.791559f, -0.59771f
                             };

    openGLSetCurBindShaderFloat2V("shadowTaps", 12, shadow_taps);


    //---------------------------------------------------------------------------
    //render to msaa_FBO
    glVerify(this->renderToFullScreenQuad());
    //---------------------------------------------------------------------------

    // reset state
    glActiveTexture(GL_TEXTURE5);
    glDisable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE3);
    glDisable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE2);
    glDisable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE1);
    glDisable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE0);
    glDisable(GL_TEXTURE_2D);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

}

void FluidRenderTask::renderToFullScreenQuad()
{
    glBegin(GL_QUADS);

    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(-1.0f, -1.0f);

    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(1.0f, -1.0f);

    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(1.0f, 1.0f);

    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(-1.0f, 1.0f);

    glEnd();
}

void FluidRenderTask::configFakeLightUniforms(bool reverse_light_dir)
{
    glm::vec3 light_pos = { 100, 150, 0 };
    glm::vec3 light_target = { 0, 0, 0 };
    glm::vec3 light_dir = glm::normalize(light_target - light_pos);
    glm::vec3 light_up = { 0.0f, 1.0f, 0.0f };

    glm::mat4 light_proj_mat = glm::perspective(glm::radians(45.0f), 1.0f, 1.0f, 1000.0f);
    glm::mat4 light_model_view_mat = glm::lookAt(light_pos, light_target, light_up);
    glm::mat4 light_transform_mat = light_proj_mat*light_model_view_mat;


    openGLSetCurBindShaderMat4("lightTransform", light_transform_mat);
    openGLSetCurBindShaderVec3("lightPos", light_pos);
    openGLSetCurBindShaderVec3("lightDir", reverse_light_dir ? -light_dir : light_dir);
    openGLSetCurBindShaderFloat("spotMin", 0.0f);
    openGLSetCurBindShaderFloat("spotMax", 0.5f);
}

void FluidRenderTask::configCameraUniforms()
{
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
    Camera<double> & camera = render_scene_config.camera();
    camera.configCameraToCurBindShader();

    //set model transform, need further consideration
    const Matrix4f & model_trans = this->transform().transformMatrix();
    openGLSetCurBindShaderMat4("model_trans", model_trans);
}

}//end of namespace Physika