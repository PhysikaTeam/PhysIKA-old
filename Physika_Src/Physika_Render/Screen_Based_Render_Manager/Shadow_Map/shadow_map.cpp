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

#include <Physika_Render/OpenGL_Primitives/glew_utilities.h>
#include <Physika_Render/OpenGL_Primitives/opengl_primitives.h>
#include <GL/freeglut.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Physika_Render/OpenGL_Shaders/shaders.h"
#include "Physika_Render/Screen_Based_Render_Manager/Shadow_Map/shadow_map.h"


namespace Physika {

ShadowMap::ShadowMap()
{
    this->shadow_program_.createFromCStyleString(vertex_shader, fragment_pass_through_shader);
    this->shadow_map_render_program_.createFromCStyleString(vertex_pass_through_shader, shadow_map_fragment_shader);
    this->initShadowFrameBuffer();
}

ShadowMap::~ShadowMap()
{
    this->destoryShadowFrameBuffer();
}

void ShadowMap::initShadowFrameBuffer()
{
    //create shadow_FBO
    glVerify(glGenFramebuffers(1, &this->shadow_FBO_));
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, this->shadow_FBO_));

    //create shadow_TEX
    glVerify(glGenTextures(1, &this->shadow_TEX_));
    glVerify(glBindTexture(GL_TEXTURE_2D, this->shadow_TEX_));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));

    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER));
    
    GLfloat border_color[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glVerify(glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color));

    //this is to allow usage of shadow2DProj function in the shader 
    //glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE));
    //glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LESS));
    //glVerify(glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY));

    //allocate memory
    glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadow_width_resolution_, shadow_height_resolution_, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL));

    //bind shadow_TEX to shadow_FBO
    glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, this->shadow_TEX_, 0));

    if (openGlCheckCurFramebufferStatus() != GL_FRAMEBUFFER_COMPLETE)
        exit(EXIT_FAILURE);

    //init depth buffer values to 1.0f
    glVerify(glClear(GL_DEPTH_BUFFER_BIT));

    //switch to default screen buffer & texture
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    glVerify(glBindTexture(GL_TEXTURE_2D, 0));
}

void ShadowMap::destoryShadowFrameBuffer()
{
    glVerify(glDeleteTextures(1, &this->shadow_TEX_));
    glVerify(glDeleteFramebuffers(1, &this->shadow_FBO_));
}

void ShadowMap::beginShadowMap()
{
    glEnable(GL_DEPTH_TEST);

    //enable polygon offset
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(8.f, 8.f);

    //switch to shadow_FBO
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, this->shadow_FBO_));

    //clear background and depth buffer
    glVerify(glClear(GL_DEPTH_BUFFER_BIT));

    //reset viewport 
    glVerify(glViewport(0, 0, shadow_width_resolution_, shadow_height_resolution_));

    // draw back faces 
    glVerify(glDisable(GL_CULL_FACE));

    //use shadow_program
    this->shadow_program_.use();

    //set shadow_program uniform data
    glVerify(glUniformMatrix4fv(glGetUniformLocation(this->shadow_program_.id(), "objectTransform"), 1, false, glm::value_ptr(glm::mat4(1.0))));
}


void ShadowMap::endShadowMap()
{
    //disable polygon offset
    glDisable(GL_POLYGON_OFFSET_FILL);

    //enable cull face
    glEnable(GL_CULL_FACE);

    //unuse shadow_program
    this->shadow_program_.unUse();
    
    //switch to default screen buffer
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

GLuint ShadowMap::shadowTexId() const
{
    return this->shadow_TEX_;
}

GLuint ShadowMap::shadowFboId() const
{
    return this->shadow_FBO_;
}

void ShadowMap::renderShadowTexToScreen()
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

    glVerify(glBindTexture(GL_TEXTURE_2D, this->shadow_TEX_));
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

}// end of namespace Physi