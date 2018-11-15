/*
* @file shadow_map.h
* @Brief class ShadowMap
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


#ifndef PHYSIKA_RENDER_SCREEN_BASED_MAP_SHADOW_RENDER_SHADOW_MAP_H
#define PHYSIKA_RENDER_SCREEN_BASED_MAP_SHADOW_RENDER_SHADOW_MAP_H

#include "Physika_Render/OpenGL_Shaders/shader_program.h"

namespace Physika{

class ShadowMap
{
public:
    ShadowMap();
    ~ShadowMap();

    //disable copy
    ShadowMap(const ShadowMap &) = delete;
    ShadowMap & operator = (const ShadowMap &) = delete;

    //enable move
    ShadowMap(ShadowMap && rhs) noexcept;
    ShadowMap & operator = (ShadowMap && rhs) noexcept;

    void beginShadowMap();
    void endShadowMap();

    unsigned int shadowMapTexId() const;
    unsigned int shadowMapFboId() const;

    void renderShadowMapToScreen();

private:
    void initShadowMapFBO();
    void destoryShadowMapFBO();

private:

    unsigned int shadow_map_TEX_ = 0;
    unsigned int shadow_map_FBO_ = 0;

    ShaderProgram shadow_map_program_;              //used to create shadow map
    ShaderProgram shadow_map_render_program_;       //used to render shadow map to screen

    const int shadow_width_resolution_ = 2048;
    const int shadow_height_resolution_ = 2048;
};

}// end of namespace Physika

#endif // PHYSIKA_RENDER_SCREEN_BASED_RENDER_SHADOW_MAP_SHADOW_MAP_H
