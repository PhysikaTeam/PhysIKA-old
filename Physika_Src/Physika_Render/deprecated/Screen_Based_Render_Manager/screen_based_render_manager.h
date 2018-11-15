/*
* @file screen_based_render_manager.h
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

#ifndef PHYSIKA_RENDER_SCREEN_BASED_RENDER_MANAGER_SCREEN_BASED_RENDER_MANAGER_H
#define PHYSIKA_RENDER_SCREEN_BASED_RENDER_MANAGER_SCREEN_BASED_RENDER_MANAGER_H

#include <vector>

#include "Physika_Render/OpenGL_Shaders/shader_program.h"
#include "Physika_Render/Screen_Based_Render_Manager/Shadow_Map/shadow_map.h"

namespace Physika{

class ShaderProgram;
class SpotLight;
class FlexSpotLight;

class ScreenBasedRenderManager 
{
public:
    ScreenBasedRenderManager();
    ~ScreenBasedRenderManager();

    //disable copy
    ScreenBasedRenderManager(const ScreenBasedRenderManager &) = delete;
    ScreenBasedRenderManager & operator = (const ScreenBasedRenderManager &) = delete;

    void enableUseShadowmap();
    void disableUseShadowmap();
    bool isUseShadowmap() const;

    void enableUseGammaCorrection();
    void disableUseGammaCorrection();
    bool isUseGammaCorrection() const;

    void enableUseHDR();
    void disableUseHDR();
    bool isUseHDR() const;

    unsigned int screenWidth() const;
    unsigned int screenHeight() const;

    void render();
    void renderAllNormalRenderTasks(bool bind_shader = true);
    void renderAllScreenBasedRenderTasks();

    void resetMsaaFBO(unsigned int screen_width, unsigned int screen_height);
    unsigned int msaaFBO() const;

private:
    void beginFrame();
    void endFrame();
    void renderToScreen();

    void createSpotLightShadowMaps();
    void setSpotLightProjAndViewMatrix(const SpotLight * spot_light);
    void configSpotLightShadowMapUnifroms(const SpotLight * spot_light, unsigned int spot_light_id, unsigned int shadow_map_tex_unit);

    void createFlexSpotLightShadowMaps();
    void setFlexSpotLightProjAndViewMatrix(const FlexSpotLight * spot_light);
    void configFlexSpotLightShadowMapUnifroms(const FlexSpotLight * spot_light, unsigned int spot_light_id, unsigned int shadow_map_tex_unit);

private:
    void initMsaaFrameBuffer();
    void destroyMsaaFrameBuffer();

    void initScreenVAO();
    void destroyScreenVAO();

private:
    unsigned int screen_width_ = 1280; //change by resetMsaaFBO()
    unsigned int screen_height_ = 720; //change by resetMsaaFBO()

    bool use_shadow_map_ = true;
    bool use_gamma_correction_ = true;
    bool use_hdr_ = false;

    int          msaa_samples_ = 8;
    unsigned int msaa_FBO_ = 0;
    unsigned int msaa_color_RBO_ = 0;
    unsigned int msaa_color_TEX_ = 0;
    unsigned int msaa_depth_RBO_ = 0;

    ShaderProgram msaa_shader_program_;
    unsigned int screen_VAO_ = 0;
    unsigned int screen_vertex_VBO_ = 0;

    //spot light shadow maps
    std::vector<ShadowMap> spot_light_shadow_maps_;
    std::vector<ShadowMap> flex_spot_light_shadow_maps_;    
};

}//end of namespace Physika


#endif //PHYSIKA_RENDER_SCREEN_BASED_RENDER_MANAGER_SCREEN_BASED_RENDER_MANAGER_H