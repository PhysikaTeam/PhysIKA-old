/*
* @file shadow_map.cpp
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

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Physika_Render/OpenGL_Primitives/glew_utilities.h>
#include <GL/freeglut.h>

#include "Physika_Render/OpenGL_Shaders/shader_srcs.h"


#include "shadow_map_shader_srcs.h"
#include "shadow_map.h"

namespace Physika {

ShadowMap::ShadowMap()
{
    this->shadow_map_program_.createFromCStyleString(shadow_map_vertex_shader, shadow_map_frag_shader);
    this->shadow_map_render_program_.createFromCStyleString(vertex_pass_through_shader, shadow_map_fragment_shader);
    this->initShadowMapFBO();
}

ShadowMap::ShadowMap(ShadowMap && rhs) noexcept
    :shadow_map_program_(std::move(rhs.shadow_map_program_)),
    shadow_map_render_program_(std::move(rhs.shadow_map_render_program_))
{
    this->shadow_map_TEX_ = rhs.shadow_map_TEX_;
    this->shadow_map_FBO_ = rhs.shadow_map_FBO_;

    rhs.shadow_map_TEX_ = 0;
    rhs.shadow_map_FBO_ = 0;
}

ShadowMap & ShadowMap::operator =(ShadowMap && rhs) noexcept
{
    this->shadow_map_program_ = std::move(rhs.shadow_map_program_);
    this->shadow_map_render_program_ = std::move(rhs.shadow_map_render_program_);

    this->shadow_map_TEX_ = rhs.shadow_map_TEX_;
    this->shadow_map_FBO_ = rhs.shadow_map_FBO_;

    rhs.shadow_map_TEX_ = 0;
    rhs.shadow_map_FBO_ = 0;

    return *this;
}

ShadowMap::~ShadowMap()
{
    this->destoryShadowMapFBO();
}

void ShadowMap::initShadowMapFBO()
{
    //create shadow_FBO
    glVerify(glGenFramebuffers(1, &this->shadow_map_FBO_));
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, this->shadow_map_FBO_));

    //create shadow_TEX
    glVerify(glGenTextures(1, &this->shadow_map_TEX_));
    glVerify(glBindTexture(GL_TEXTURE_2D, this->shadow_map_TEX_));

    //set filter & wrap
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER));
    
    //set border color
    GLfloat border_color[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glVerify(glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color));

    //allocate memory
    glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadow_width_resolution_, shadow_height_resolution_, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL));

    //bind shadow_TEX to shadow_FBO
    glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, this->shadow_map_TEX_, 0));

    if (openGLCheckCurFramebufferStatus() != GL_FRAMEBUFFER_COMPLETE)
        exit(EXIT_FAILURE);

    //switch to default screen buffer & texture
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    glVerify(glBindTexture(GL_TEXTURE_2D, 0));
}

void ShadowMap::destoryShadowMapFBO()
{
    glVerify(glDeleteTextures(1, &this->shadow_map_TEX_));
    glVerify(glDeleteFramebuffers(1, &this->shadow_map_FBO_));
}

void ShadowMap::beginShadowMap()
{
    //switch to shadow_FBO & reset viewport
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, this->shadow_map_FBO_));
    glVerify(glViewport(0, 0, shadow_width_resolution_, shadow_height_resolution_));

    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glVerify(glEnable(GL_DEPTH_TEST));
    glVerify(glDisable(GL_CULL_FACE));

    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(8.f, 8.f);
    
    //clear depth buffer
    glVerify(glClear(GL_DEPTH_BUFFER_BIT));

    //use shadow_program
    this->shadow_map_program_.use();
}


void ShadowMap::endShadowMap()
{
    //unuse shadow_program
    this->shadow_map_program_.unUse();
    
    glDisable(GL_POLYGON_OFFSET_FILL);
    glEnable(GL_CULL_FACE);
 
    glVerify(glPopAttrib());

    //switch to default screen buffer
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

unsigned int ShadowMap::shadowMapTexId() const
{
    return this->shadow_map_TEX_;
}

unsigned int ShadowMap::shadowMapFboId() const
{
    return this->shadow_map_FBO_;
}

void ShadowMap::renderShadowMapToScreen()
{
    glVerify(glPushAttrib(GL_ALL_ATTRIB_BITS));

    this->shadow_map_render_program_.use();
    
    glVerify(glClearColor(0.0f, 0.0f, 0.0f, 0.0f));
    glVerify(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
    

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glVerify(glBindTexture(GL_TEXTURE_2D, this->shadow_map_TEX_));
    openGLRenderToFullScreenQuad();
    glVerify(glBindTexture(GL_TEXTURE_2D, 0));

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    this->shadow_map_render_program_.unUse();

    glVerify(glPopAttrib());

    glVerify(glFinish());
    glVerify(glutSwapBuffers());
}

}// end of namespace Physika