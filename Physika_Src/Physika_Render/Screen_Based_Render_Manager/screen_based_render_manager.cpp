/*
* @file screen_based_render_manager.cpp
* @Brief class ScreenBasedRenderManager
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

#include <cmath>

#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"
#include <GL/freeglut.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Vectors/vector_4d.h"

#include "Physika_Render/Color/color.h"
#include "Physika_Render/Lights/light_base.h"
#include "Physika_Render/Lights/spot_light.h"
#include "Physika_Render/Lights/flex_spot_light.h"
#include "Physika_Render/Render_Task_Base/render_task_base.h"
#include "Physika_Render/Render_Scene_Config/render_scene_config.h"
#include "Physika_Render/Global_Unifrom_Config/global_uniform_config.h"

#include "screen_based_render_manager.h"
#include "screen_based_shader_srcs.h"

namespace Physika {

ScreenBasedRenderManager::ScreenBasedRenderManager()
{
    this->msaa_shader_program_.createFromCStyleString(screen_based_vertex_shader, screen_based_frag_shader);
    this->initMsaaFrameBuffer();
    this->initScreenVAO();
}

ScreenBasedRenderManager::~ScreenBasedRenderManager()
{
    this->destroyMsaaFrameBuffer();
    this->destroyScreenVAO();
}

void ScreenBasedRenderManager::enableUseShadowmap()
{
    use_shadow_map_ = true;
}

void ScreenBasedRenderManager::disableUseShadowmap()
{
    use_shadow_map_ = false;
}

bool ScreenBasedRenderManager::isUseShadowmap() const
{
    return use_shadow_map_;
}

void ScreenBasedRenderManager::enableUseGammaCorrection()
{
    use_gamma_correction_ = true;
}

void ScreenBasedRenderManager::disableUseGammaCorrection()
{
    use_gamma_correction_ = false;
}

bool ScreenBasedRenderManager::isUseGammaCorrection() const
{
    return use_gamma_correction_;
}

void ScreenBasedRenderManager::enableUseHDR()
{
    use_hdr_ = true;
}

void ScreenBasedRenderManager::disableUseHDR()
{
    use_hdr_ = false;
}

bool ScreenBasedRenderManager::isUseHDR() const
{
    return use_hdr_;
}

unsigned int ScreenBasedRenderManager::screenWidth() const
{
    return screen_width_;
}

unsigned int ScreenBasedRenderManager::screenHeight() const
{
    return screen_height_;
}

void ScreenBasedRenderManager::render()
{
    //create shadow map for spot lights
    if(use_shadow_map_)
    {
        this->createSpotLightShadowMaps();
        this->createFlexSpotLightShadowMaps();
    }
    
    GlobalUniformConfig & global_uniform_config = GlobalUniformConfig::getSingleton();
    global_uniform_config.setBool("use_shadow_map", use_shadow_map_);
    
    this->beginFrame();

    this->renderAllNormalRenderTasks();
    this->renderAllScreenBasedRenderTasks();

    this->endFrame();

    this->renderToScreen();

    //switch to default screen buffer to render depth buffer
    //glVerify(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    //glVerify(glViewport(0, 0, this->screen_width_, this->screen_height_));
    //this->spot_light_shadow_maps_[0].renderShadowMapToScreen();
}

void ScreenBasedRenderManager::resetMsaaFBO(unsigned int screen_width, unsigned int screen_height)
{
    if (this->screen_width_ == screen_width && this->screen_height_ == screen_height)
        return;

    this->screen_width_ = screen_width;
    this->screen_height_ = screen_height;

    this->destroyMsaaFrameBuffer();
    this->initMsaaFrameBuffer();
}

unsigned int ScreenBasedRenderManager::msaaFBO() const
{
    return this->msaa_FBO_;
}

void ScreenBasedRenderManager::renderAllNormalRenderTasks(bool bind_shader)
{
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();

    for (unsigned int i = 0; i < render_scene_config.numRenderTasks(); ++i)
    {
        std::shared_ptr<RenderTaskBase> render_task = render_scene_config.getRenderTaskAtIndex(i);
        if (render_task->type() != RenderTaskType::NORMAL_RENDER_TASK)
            continue;

        if (bind_shader == true)
            render_task->enableBindShader();
        else
            render_task->disableBindShader();

        render_task->renderTask();
    }
}

void ScreenBasedRenderManager::renderAllScreenBasedRenderTasks()
{
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();

    for (unsigned int i = 0; i < render_scene_config.numRenderTasks(); ++i)
    {
        std::shared_ptr<RenderTaskBase> render_task = render_scene_config.getRenderTaskAtIndex(i);
        if (render_task->type() != RenderTaskType::SCREEN_BASED_RENDER_TASK)
            continue;

        render_task->renderTask();
    }
}

void ScreenBasedRenderManager::beginFrame()
{
    float window_clear_color[3];
    glGetFloatv(GL_COLOR_CLEAR_VALUE, window_clear_color);

    //switch to msaa_FBO & set view port
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, this->msaa_FBO_));
    glVerify(glViewport(0, 0, this->screen_width_, this->screen_height_));

    glVerify(glPushAttrib(GL_ALL_ATTRIB_BITS));

    glVerify(glEnable(GL_DEPTH_TEST));
    glVerify(glClearColor(window_clear_color[0], window_clear_color[1], window_clear_color[2], 1.0f));
    glVerify(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
}

void ScreenBasedRenderManager::endFrame()
{
    glVerify(glPopAttrib());

    //clear all global uniform configs
    GlobalUniformConfig & global_uniform_config = GlobalUniformConfig::getSingleton();
    global_uniform_config.clear();

    /*
     
    //render to default frame buffer

    //specify msaa_FBO to read and default FBO to draw
    glVerify(glBindFramebuffer(GL_READ_FRAMEBUFFER, this->msaa_FBO_));
    glVerify(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));

    //blit the msaa_FBO to the window(default FBO), i.e. render to the current window
    glVerify(glBlitFramebuffer(0, 0, this->screen_width_, this->screen_height_, 0, 0, this->screen_width_, this->screen_height_, GL_COLOR_BUFFER_BIT, GL_LINEAR));

    //render help to back buffer
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, 0));

    */
}

void ScreenBasedRenderManager::renderToScreen()
{
    //switch back to the default frame buffer
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    glVerify(glViewport(0, 0, this->screen_width_, this->screen_height_));

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    this->msaa_shader_program_.use();
    openGLSetCurBindShaderBool("use_gamma_correction", use_gamma_correction_);
    openGLSetCurBindShaderBool("use_hdr", use_hdr_);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, msaa_color_TEX_);

    glBindVertexArray(screen_VAO_);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    glBindTexture(GL_TEXTURE_2D, 0);

    this->msaa_shader_program_.unUse();
}

void ScreenBasedRenderManager::createSpotLightShadowMaps()
{
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();

    unsigned int cur_spot_light_id = 0;
    for(unsigned int i = 0; i < render_scene_config.numLights(); ++i)
    {
        const std::shared_ptr<LightBase> & light = render_scene_config.lightAtIndex(i);
        if (light->type() != LightType::SPOT_LIGHT || light->isEnableLighting() == false)
            continue;

        //add new default shadow map for spot light
        if (cur_spot_light_id == spot_light_shadow_maps_.size())
            this->spot_light_shadow_maps_.emplace_back();
            
        ShadowMap & shadow_map = this->spot_light_shadow_maps_[cur_spot_light_id];
        SpotLight * spot_light = static_cast<SpotLight *>(light.get());
        unsigned int shadow_map_tex_id = shadow_map.shadowMapTexId();

        shadow_map.beginShadowMap();

        //config light projection & view matrix
        this->setSpotLightProjAndViewMatrix(spot_light);

        this->renderAllNormalRenderTasks(false);

        shadow_map.endShadowMap();

        //bind shadow map to texture unit
        glActiveTexture(GL_TEXTURE1 + cur_spot_light_id); //start from one
        glBindTexture(GL_TEXTURE_2D, shadow_map_tex_id);
        glActiveTexture(GL_TEXTURE0);

        //config shadow map to uniforms
        unsigned int shadow_map_tex_unit = cur_spot_light_id + 1;
        this->configSpotLightShadowMapUnifroms(spot_light, cur_spot_light_id, shadow_map_tex_unit);

        ++cur_spot_light_id;
    }
}

void ScreenBasedRenderManager::createFlexSpotLightShadowMaps()
{
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();

    unsigned int cur_flex_spot_light_id = 0;
    for (unsigned int i = 0; i < render_scene_config.numLights(); ++i)
    {
        const std::shared_ptr<LightBase> & light = render_scene_config.lightAtIndex(i);
        if (light->type() != LightType::FLEX_SPOT_LIGHT || light->isEnableLighting() == false)
            continue;

        //add new default shadow map for spot light
        if (cur_flex_spot_light_id == flex_spot_light_shadow_maps_.size())
            this->flex_spot_light_shadow_maps_.emplace_back();

        ShadowMap & shadow_map = this->flex_spot_light_shadow_maps_[cur_flex_spot_light_id];
        FlexSpotLight * flex_spot_light = static_cast<FlexSpotLight *>(light.get());
        unsigned int shadow_map_tex_id = shadow_map.shadowMapTexId();

        shadow_map.beginShadowMap();

        //config light projection & view matrix
        this->setFlexSpotLightProjAndViewMatrix(flex_spot_light);

        this->renderAllNormalRenderTasks(false);

        shadow_map.endShadowMap();

        //bind shadow map to texture unit
        glActiveTexture(GL_TEXTURE1 + this->spot_light_shadow_maps_.size() + cur_flex_spot_light_id);
        glBindTexture(GL_TEXTURE_2D, shadow_map_tex_id);
        glActiveTexture(GL_TEXTURE0);

        //config shadow map to uniforms
        unsigned int shadow_map_tex_unit = this->spot_light_shadow_maps_.size() + cur_flex_spot_light_id + 1;
        this->configFlexSpotLightShadowMapUnifroms(flex_spot_light, cur_flex_spot_light_id, shadow_map_tex_unit);

        ++cur_flex_spot_light_id;
    }
}

void ScreenBasedRenderManager::setSpotLightProjAndViewMatrix(const SpotLight * spot_light)
{
    glm::mat4 light_proj_mat = spot_light->lightProjMatrix();
    glm::mat4 light_view_mat = spot_light->lightViewMatrix();

    openGLSetCurBindShaderMat4("light_proj_trans", light_proj_mat);
    openGLSetCurBindShaderMat4("light_view_trans", light_view_mat);
}

void ScreenBasedRenderManager::configSpotLightShadowMapUnifroms(const SpotLight * spot_light, unsigned int spot_light_id, unsigned int shadow_map_tex_unit)
{
    GlobalUniformConfig & global_uniform_config = GlobalUniformConfig::getSingleton();
    global_uniform_config.setBool("use_shadow_map", true);

    std::string shadow_map_str = "spot_light_shadow_maps[" + std::to_string(spot_light_id) + "]";
    global_uniform_config.setInt(shadow_map_str + ".shadow_map_tex", shadow_map_tex_unit);
    global_uniform_config.setBool(shadow_map_str + ".has_shadow_map", true);
    
}

void ScreenBasedRenderManager::setFlexSpotLightProjAndViewMatrix(const FlexSpotLight * flex_spot_light)
{
    glm::mat4 light_proj_mat = flex_spot_light->lightProjMatrix();
    glm::mat4 light_view_mat = flex_spot_light->lightViewMatrix();

    openGLSetCurBindShaderMat4("light_proj_trans", light_proj_mat);
    openGLSetCurBindShaderMat4("light_view_trans", light_view_mat);
}

void ScreenBasedRenderManager::configFlexSpotLightShadowMapUnifroms(const FlexSpotLight * flex_spot_light, unsigned int flex_spot_light_id, unsigned int shadow_map_tex_unit)
{
    GlobalUniformConfig & global_uniform_config = GlobalUniformConfig::getSingleton();
    global_uniform_config.setBool("use_shadow_map", true);

    std::string shadow_map_str = "flex_spot_light_shadow_maps[" + std::to_string(flex_spot_light_id) + "]";
    global_uniform_config.setInt(shadow_map_str + ".shadow_map_tex", shadow_map_tex_unit);
    global_uniform_config.setBool(shadow_map_str + ".has_shadow_map", true);
}

void ScreenBasedRenderManager::initMsaaFrameBuffer()
{
    //determine msaa_samples
    int samples;
    glGetIntegerv(GL_MAX_SAMPLES_EXT, &samples);
    samples = std::min({ samples, this->msaa_samples_, 4 }); // clamp samples to 4

                                                             //create msaa_FBO
    glVerify(glGenFramebuffers(1, &this->msaa_FBO_));
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, this->msaa_FBO_));

    //create msaa_color_RBO & bind it to msaa_FBO
    //glVerify(glGenRenderbuffers(1, &this->msaa_color_RBO_));
    //glVerify(glBindRenderbuffer(GL_RENDERBUFFER, this->msaa_color_RBO_));
    //glVerify(glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_RGBA8, this->screen_width_, this->screen_height_));
    //glVerify(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, this->msaa_color_RBO_));
    //glVerify(glBindRenderbuffer(GL_RENDERBUFFER, 0));

    glVerify(glGenTextures(1, &this->msaa_color_TEX_));
    glVerify(glBindTexture(GL_TEXTURE_2D, this->msaa_color_TEX_));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, this->screen_width_, this->screen_height_, 0, GL_RGB, GL_FLOAT, nullptr));
    glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->msaa_color_TEX_, 0));
    glVerify(glBindTexture(GL_TEXTURE_2D, 0));

    //create msaa_depth_RBO & bind it to msaa_FBO
    glVerify(glGenRenderbuffers(1, &this->msaa_depth_RBO_));
    glVerify(glBindRenderbuffer(GL_RENDERBUFFER, this->msaa_depth_RBO_));
    glVerify(glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_DEPTH_COMPONENT, this->screen_width_, this->screen_height_));
    glVerify(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, this->msaa_depth_RBO_));
    glVerify(glBindRenderbuffer(GL_RENDERBUFFER, 0));

    //check frame buffer status
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cerr << "error: msaa_FBO is incomplete !" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    glEnable(GL_MULTISAMPLE);

    //switch to default framebuffer
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

void ScreenBasedRenderManager::destroyMsaaFrameBuffer()
{
    glVerify(glDeleteFramebuffers(1, &this->msaa_FBO_));
    glVerify(glDeleteRenderbuffers(1, &this->msaa_color_RBO_));
    glVerify(glDeleteRenderbuffers(1, &this->msaa_depth_RBO_));
    glVerify(glDeleteTextures(1, &this->msaa_color_TEX_));
}

void ScreenBasedRenderManager::initScreenVAO()
{
    float screen_vertices[] = {
                                // positions        //tex coords
                                -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
                                -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
                                 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,

                                 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
                                -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
                                 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
                               };

    // setup plane VAO
    glGenVertexArrays(1, &screen_VAO_);
    glBindVertexArray(screen_VAO_);

    glGenBuffers(1, &screen_vertex_VBO_);
    glBindBuffer(GL_ARRAY_BUFFER, screen_vertex_VBO_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(screen_vertices), &screen_vertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);
}

void ScreenBasedRenderManager::destroyScreenVAO()
{
    glDeleteVertexArrays(1, &screen_VAO_);
    glDeleteBuffers(1, &screen_vertex_VBO_);
}

}//end of namespace Physika
