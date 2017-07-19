/*
* @file fluid_render.cpp
* @Brief render for fluid using Screen-Based method
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

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include "Physika_Render/OpenGL_Shaders/shaders.h"

#include "Physika_GUI/Glut_Window/glut_window.h"
#include "Physika_Render/Screen_Based_Render_Manager/screen_based_render_manager.h"
#include "Physika_Render/Screen_Based_Render_Manager/Fluid_Render/fluid_render.h"

namespace Physika{

FluidRender::FluidRender(unsigned int fluid_particle_num, unsigned int diffuse_particle_num, unsigned int screen_width, unsigned int screen_height)
{
    this->initFluidParticleBuffer(fluid_particle_num);
    this->initDiffuseParticleBuffer(diffuse_particle_num);

    this->screen_width_ = screen_width;
    this->screen_height_ = screen_height;

    this->initFrameBufferAndShaderProgram();

}

FluidRender::~FluidRender()
{
    this->destroyFluidParticleBuffer();
    this->destroyDiffusePartcileBuffer();
    this->destroyFrameBufferAndShaderProgram();
}

void FluidRender::initFluidParticleBuffer(unsigned int fluid_particle_num)
{
    fluid_particle_buffer_.fluid_particle_num = fluid_particle_num;

    // create position_VBO
    glVerify(glGenBuffers(1, &fluid_particle_buffer_.position_VBO_));
    glVerify(glBindBuffer(GL_ARRAY_BUFFER, fluid_particle_buffer_.position_VBO_));
    glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * fluid_particle_num, 0, GL_DYNAMIC_DRAW));

    // create density_VBO
    glVerify(glGenBuffers(1, &fluid_particle_buffer_.density_VBO_));
    glVerify(glBindBuffer(GL_ARRAY_BUFFER, fluid_particle_buffer_.density_VBO_));
    glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float)*fluid_particle_num, 0, GL_DYNAMIC_DRAW));

    // create anisotropy_VBO
    for (int i = 0; i < 3; ++i)
    {
        glVerify(glGenBuffers(1, &fluid_particle_buffer_.anisotropy_VBO_[i]));
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, fluid_particle_buffer_.anisotropy_VBO_[i]));
        glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * fluid_particle_num, 0, GL_DYNAMIC_DRAW));
    }

    // create indices_EBO
    glVerify(glGenBuffers(1, &fluid_particle_buffer_.indices_EBO_));
    glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, fluid_particle_buffer_.indices_EBO_));
    glVerify(glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int)*fluid_particle_num, 0, GL_DYNAMIC_DRAW));
}

void FluidRender::initDiffuseParticleBuffer(unsigned int diffuse_partcle_num)
{
    diffuse_particle_buffer_.diffuse_particle_num = diffuse_partcle_num;

    if (diffuse_partcle_num > 0)
    {
        //create diffuse_position_VBO
        glVerify(glGenBuffers(1, &diffuse_particle_buffer_.diffuse_position_VBO_));
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, diffuse_particle_buffer_.diffuse_position_VBO_));
        glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * diffuse_partcle_num, 0, GL_DYNAMIC_DRAW));
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));

        //create diffuse_velocity_VBO
        glVerify(glGenBuffers(1, &diffuse_particle_buffer_.diffuse_velocity_VBO_));
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, diffuse_particle_buffer_.diffuse_velocity_VBO_));
        glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * diffuse_partcle_num, 0, GL_DYNAMIC_DRAW));
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));

        //create diffuse_indices_EBO
        glVerify(glGenBuffers(1, &diffuse_particle_buffer_.diffuse_indices_EBO_));
        glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, diffuse_particle_buffer_.diffuse_indices_EBO_));
        glVerify(glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * diffuse_partcle_num, 0, GL_DYNAMIC_DRAW));
        glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
    }
}

void FluidRender::initFrameBufferAndShaderProgram()
{
    //create depth_TEX
    glVerify(glGenTextures(1, &this->depth_TEX_));
    glVerify(glBindTexture(GL_TEXTURE_RECTANGLE, this->depth_TEX_));

    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    glVerify(glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_LUMINANCE32F_ARB, this->screen_width_, this->screen_height_, 0, GL_LUMINANCE, GL_FLOAT, NULL));

    //create depth_smooth_TEX
    glVerify(glGenTextures(1, &this->depth_smooth_TEX_));
    glVerify(glBindTexture(GL_TEXTURE_2D, this->depth_smooth_TEX_));

    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, this->screen_width_, this->screen_height_, 0, GL_LUMINANCE, GL_FLOAT, NULL));

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //create scene_TEX
    glVerify(glGenTextures(1, &this->scene_TEX_));
    glVerify(glBindTexture(GL_TEXTURE_2D, this->scene_TEX_));

    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, this->screen_width_, this->screen_height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));

    //create scene_FBO & bind secene_TEX to scene_FBO
    glVerify(glGenFramebuffers(1, &this->scene_FBO_));
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, this->scene_FBO_));
    glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->scene_TEX_, 0));

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //create depth_FBO & bind depth_TEX to depth_FBO
    glVerify(glGenFramebuffers(1, &this->depth_FBO_));
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, this->depth_FBO_));
    glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_ARB, this->depth_TEX_, 0));

    //need further consideration
    //create zbuffer_RBO & bind it to depth_FBO
    GLuint zbuffer_RBO;
    glVerify(glGenRenderbuffers(1, &zbuffer_RBO));
    glVerify(glBindRenderbuffer(GL_RENDERBUFFER, zbuffer_RBO));
    glVerify(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, this->screen_width_, this->screen_height_));
    glVerify(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, zbuffer_RBO));

    //specify which buffer to draw and read
    glVerify(glDrawBuffer(GL_COLOR_ATTACHMENT0));
    glVerify(glReadBuffer(GL_COLOR_ATTACHMENT0));

    //check frame buffer status
    glVerify(glCheckFramebufferStatus(GL_FRAMEBUFFER));

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //create reflect_TEX
    glVerify(glGenTextures(1, &this->reflect_TEX_));
    glVerify(glBindTexture(GL_TEXTURE_2D, this->reflect_TEX_));

    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, this->screen_width_, this->screen_height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));

    //create thickness_TEX
    glVerify(glGenTextures(1, &this->thickness_TEX_));
    glVerify(glBindTexture(GL_TEXTURE_2D, this->thickness_TEX_));

    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, this->screen_width_, this->screen_height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));


    //create thickness_FBO and bind thickness_TEX to it
    glVerify(glGenFramebuffers(1, &this->thickness_FBO_));
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, this->thickness_FBO_));
    glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->thickness_TEX_, 0));

    //create thick_zuffer_RBO and bind it to thickness_FBO
    GLuint thick_zuffer_RBO;
    glVerify(glGenRenderbuffers(1, &thick_zuffer_RBO));
    glVerify(glBindRenderbuffer(GL_RENDERBUFFER, thick_zuffer_RBO));
    glVerify(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, this->screen_width_, this->screen_height_));
    glVerify(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, thick_zuffer_RBO));

    //check frame buffer status
    glVerify(glCheckFramebufferStatus(GL_FRAMEBUFFER));

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //create shader programs

    //diffuse_program
    this->diffuse_program_.createFromCStyleString(vertex_diffuse_shader, fragment_diffuse_shader, geometry_diffuse_shader);
    glVerify(glProgramParameteriEXT(this->diffuse_program_.id(), GL_GEOMETRY_VERTICES_OUT_EXT, 4));
    glVerify(glProgramParameteriEXT(this->diffuse_program_.id(), GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS));
    glVerify(glProgramParameteriEXT(this->diffuse_program_.id(), GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP));
    glVerify(glLinkProgram(this->diffuse_program_.id()));

    //point_thickness_program
    this->point_thickness_program_.createFromCStyleString(vertex_point_depth_shader, fragment_point_thickness_shader);
    
    //this->ellipsoid_thickness_program_.createFromCStyleString(vertex_ellipsoid_depth_shader, fragment_ellipsoid_thickness_shader);

    //ellipsoid_thickness_program
    this->ellipsoid_depth_program_.createFromCStyleString(vertex_ellipsoid_depth_shader, fragment_ellipsoid_depth_shader, geometry_ellipsoid_depth_shader);
    glVerify(glProgramParameteriEXT(this->ellipsoid_depth_program_.id(), GL_GEOMETRY_VERTICES_OUT_EXT, 4));
    glVerify(glProgramParameteriEXT(this->ellipsoid_depth_program_.id(), GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS));
    glVerify(glProgramParameteriEXT(this->ellipsoid_depth_program_.id(), GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP));
    glVerify(glLinkProgram(this->ellipsoid_depth_program_.id()));
    
    //composite_program
    this->composite_program_.createFromCStyleString(vertex_pass_through_shader, fragment_composite_shader);

    //depth_blur_program
    this->depth_blur_program_.createFromCStyleString(vertex_pass_through_shader, fragment_blur_depth_shader);

}

void FluidRender::destroyFluidParticleBuffer()
{
    glDeleteBuffers(1, &fluid_particle_buffer_.position_VBO_);
    glDeleteBuffers(1, &fluid_particle_buffer_.density_VBO_);
    glDeleteBuffers(3, fluid_particle_buffer_.anisotropy_VBO_);
    glDeleteBuffers(1, &fluid_particle_buffer_.indices_EBO_);
}

void FluidRender::destroyDiffusePartcileBuffer()
{
    if (diffuse_particle_buffer_.diffuse_particle_num > 0)
    {
        glDeleteBuffers(1, &diffuse_particle_buffer_.diffuse_position_VBO_);
        glDeleteBuffers(1, &diffuse_particle_buffer_.diffuse_velocity_VBO_);
        glDeleteBuffers(1, &diffuse_particle_buffer_.diffuse_indices_EBO_);
    }
}

void FluidRender::destroyFrameBufferAndShaderProgram()
{
    glVerify(glDeleteFramebuffers(1, &this->depth_FBO_));
    glVerify(glDeleteTextures(1, &this->depth_TEX_));
    glVerify(glDeleteTextures(1, &this->depth_smooth_TEX_));

    glVerify(glDeleteFramebuffers(1, &this->scene_FBO_));
    glVerify(glDeleteTextures(1, &this->scene_TEX_));

    glVerify(glDeleteFramebuffers(1, &this->thickness_FBO_));
    glVerify(glDeleteTextures(1, &this->thickness_TEX_));

    this->diffuse_program_.destory();
    this->point_thickness_program_.destory();
    this->ellipsoid_thickness_program_.destory();
    this->ellipsoid_depth_program_.destory();
    this->composite_program_.destory();
    this->depth_blur_program_.destory();

}

void FluidRender::updateFluidParticleBuffer(GLfloat * position_buffer, 
                                            GLfloat * density_buffer, 
                                            GLfloat * anisotropy_buffer_0, 
                                            GLfloat * anisotropy_buffer_1, 
                                            GLfloat * anisotropy_buffer_2, 
                                            GLuint * indices_buffer, 
                                            unsigned int indices_num)
{
    // regular particles

    unsigned int position_buffer_size = fluid_particle_buffer_.fluid_particle_num * 4 * sizeof(float);
    unsigned int anisotropy_buffer_size = position_buffer_size;

    unsigned int density_buffer_size = fluid_particle_buffer_.fluid_particle_num * sizeof(float);

    unsigned int incides_buffer_size = indices_num * sizeof(int);

    //update position_VBO
    glVerify(glBindBuffer(GL_ARRAY_BUFFER, fluid_particle_buffer_.position_VBO_));
    glVerify(glBufferSubData(GL_ARRAY_BUFFER, 0, position_buffer_size, position_buffer));

    //update density_VBO
    if (density_buffer)
    {
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, fluid_particle_buffer_.density_VBO_));
        glVerify(glBufferSubData(GL_ARRAY_BUFFER, 0, density_buffer_size, density_buffer));
    }

    //update anisotropy_buffer_VBO
    if (anisotropy_buffer_0)
    {
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, fluid_particle_buffer_.anisotropy_VBO_[0]));
        glVerify(glBufferSubData(GL_ARRAY_BUFFER, 0, anisotropy_buffer_size, anisotropy_buffer_0));
    }

    if (anisotropy_buffer_1)
    {
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, fluid_particle_buffer_.anisotropy_VBO_[1]));
        glVerify(glBufferSubData(GL_ARRAY_BUFFER, 0, anisotropy_buffer_size, anisotropy_buffer_1));
    }

    if (anisotropy_buffer_2)
    {
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, fluid_particle_buffer_.anisotropy_VBO_[2]));
        glVerify(glBufferSubData(GL_ARRAY_BUFFER, 0, anisotropy_buffer_size, anisotropy_buffer_2));
    }

    
    //update indices_EBO
    if (indices_buffer)
    {
        glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, fluid_particle_buffer_.indices_EBO_));
        glVerify(glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, incides_buffer_size, indices_buffer));
    }

    // reset
    glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
    glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

void FluidRender::updateDiffuseParticleBuffer(GLfloat * diffuse_position_buffer,
                                              GLfloat * diffuse_velocity_buffer,
                                              GLuint  * diffuse_indices_buffer)
{
    if (diffuse_particle_buffer_.diffuse_particle_num)
    {
        unsigned int position_buffer_size = diffuse_particle_buffer_.diffuse_particle_num * 4 * sizeof(float); //4*n*sizeof(float)
        unsigned int velocity_buffer_size = position_buffer_size; //4*n*sizeof(float)

        unsigned int indices_buffer_size = diffuse_particle_buffer_.diffuse_particle_num * sizeof(int);

        //update position_VBO
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, diffuse_particle_buffer_.diffuse_position_VBO_));
        glVerify(glBufferSubData(GL_ARRAY_BUFFER, 0, position_buffer_size, diffuse_position_buffer));
        
        //update velocity_VBO
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, diffuse_particle_buffer_.diffuse_velocity_VBO_));
        glVerify(glBufferSubData(GL_ARRAY_BUFFER, 0, velocity_buffer_size, diffuse_velocity_buffer));

        //update indices_EBO
        glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, diffuse_particle_buffer_.diffuse_indices_EBO_));
        glVerify(glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, indices_buffer_size, diffuse_indices_buffer));
    }
}



void FluidRender::renderFluidParticle(ScreenBasedRenderManager * screen_based_render, GLuint shadow_map_TEX)
{
    /********************************************************************************************************************/
    
    unsigned int screen_width = screen_based_render->glutWindow()->width();
    unsigned int screen_height = screen_based_render->glutWindow()->height();
    float        screen_aspect = static_cast<float>(screen_width) / screen_height;

    GLuint msaa_FBO = screen_based_render->msaaFBO();

    // resolve msaa back buffer to texture
    glVerify(glBindFramebuffer(GL_READ_FRAMEBUFFER_EXT, msaa_FBO));
    glVerify(glBindFramebuffer(GL_DRAW_FRAMEBUFFER_EXT, this->scene_FBO_));
    glVerify(glBlitFramebuffer(0, 0, screen_width, screen_height, 0, 0, screen_width, screen_height, GL_COLOR_BUFFER_BIT, GL_LINEAR));
    
    /********************************************************************************************************************/
   
    //switch to thickness_FBO
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, this->thickness_FBO_));
    glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, thickness_TEX_, 0));
    glVerify(glDrawBuffer(GL_COLOR_ATTACHMENT0));

    glViewport(0, 0, screen_width, screen_height);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_DEPTH_BUFFER_BIT);

    glDepthMask(GL_TRUE);
    glDisable(GL_CULL_FACE);

    screen_based_render->drawShapes();

    glClear(GL_COLOR_BUFFER_BIT);

    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    glEnable(GL_POINT_SPRITE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);

    glDepthMask(GL_FALSE);

    // make sprites larger to get smoother thickness texture
    const float thickness_scale = 4.0f;

    this->point_thickness_program_.use();
    glUniform1f(glGetUniformLocation(this->point_thickness_program_.id(), "pointRadius"), thickness_scale*this->radius_);
    glUniform1f(glGetUniformLocation(this->point_thickness_program_.id(), "pointScale"), screen_width / screen_aspect * (1.0f / (tanf(this->fov_*0.5f))));

    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, this->fluid_particle_buffer_.position_VBO_);
    glVertexPointer(3, GL_FLOAT, sizeof(float) * 4, 0);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //need further consideration about fluid_particle_num
    glDrawArrays(GL_POINTS, 0, this->fluid_particle_buffer_.fluid_particle_num);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    this->point_thickness_program_.unUse();

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_POINT_SPRITE);
    glDisable(GL_BLEND);

    /********************************************************************************************************************/

    //switch to depth_FBO
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, this->depth_FBO_));
    glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_ARB, this->depth_TEX_, 0));
    glVerify(glDrawBuffer(GL_COLOR_ATTACHMENT0));

    // draw points
    //glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    glDisable(GL_POINT_SPRITE);
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

    glViewport(0, 0, screen_width, screen_height);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    this->ellipsoid_depth_program_.use();

    glm::vec3    inv_viewport = { 1.0f / screen_width, 1.0f / screen_height, 1.0f };
    glVerify(glUniform3fv(glGetUniformLocation(this->ellipsoid_depth_program_.id(), "invViewport"), 1, glm::value_ptr(inv_viewport)));

    //std::cout << "inv_viewport: " << inv_viewport[0] << " " << inv_viewport[1] << " " << inv_viewport[2] << std::endl;

    const float view_height = tanf(this->fov_ / 2.0f);
    glm::vec3 inv_projection = { screen_aspect*view_height, view_height, 1.0f };
    glVerify(glUniform3fv(glGetUniformLocation(this->ellipsoid_depth_program_.id(), "invProjection"), 1, glm::value_ptr(inv_projection)));

    //std::cout << "inv_projection: " << inv_projection[0] << " " << inv_projection[1] << " " << inv_projection[2] << std::endl;

    glVerify(glEnableClientState(GL_VERTEX_ARRAY));
    glVerify(glBindBuffer(GL_ARRAY_BUFFER, this->fluid_particle_buffer_.position_VBO_));
    glVerify(glVertexPointer(3, GL_FLOAT, sizeof(float) * 4, 0));

    // ellipsoid eigenvectors
    
    int s1 = glGetAttribLocation(this->ellipsoid_depth_program_.id(), "q1");
    std::cout << "s1: " << s1 << std::endl;
    glVerify(glEnableVertexAttribArray(s1));
    glVerify(glBindBuffer(GL_ARRAY_BUFFER, this->fluid_particle_buffer_.anisotropy_VBO_[0]));
    glVerify(glVertexAttribPointer(s1, 4, GL_FLOAT, GL_FALSE, 0, 0));

    int s2 = glGetAttribLocation(this->ellipsoid_depth_program_.id(), "q2");
    std::cout << "s2: " << s2 << std::endl;
    glVerify(glEnableVertexAttribArray(s2));
    glVerify(glBindBuffer(GL_ARRAY_BUFFER, this->fluid_particle_buffer_.anisotropy_VBO_[1]));
    glVerify(glVertexAttribPointer(s2, 4, GL_FLOAT, GL_FALSE, 0, 0));

    int s3 = glGetAttribLocation(this->ellipsoid_depth_program_.id(), "q3");
    std::cout << "s3: " << s3 << std::endl;
    glVerify(glEnableVertexAttribArray(s3));
    glVerify(glBindBuffer(GL_ARRAY_BUFFER, this->fluid_particle_buffer_.anisotropy_VBO_[2]));
    glVerify(glVertexAttribPointer(s3, 4, GL_FLOAT, GL_FALSE, 0, 0));
    

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //need further consideration about fluid_particle_num
    glVerify(glDrawArrays(GL_POINTS, 0, this->fluid_particle_buffer_.fluid_particle_num));
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    this->ellipsoid_depth_program_.unUse();

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableVertexAttribArray(s1);
    glDisableVertexAttribArray(s2);
    glDisableVertexAttribArray(s3);

    glDisable(GL_POINT_SPRITE);

    /********************************************************************************************************************/
    // blur

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->depth_smooth_TEX_, 0));
    
    this->depth_blur_program_.use();

    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_RECTANGLE);
    glBindTexture(GL_TEXTURE_RECTANGLE, this->depth_TEX_);

    glActiveTexture(GL_TEXTURE1);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, this->thickness_TEX_);

    glVerify(glUniform1f(glGetUniformLocation(this->depth_blur_program_.id(), "blurRadiusWorld"), this->radius_*0.5f));	// blur half the radius by default
    glVerify(glUniform1f(glGetUniformLocation(this->depth_blur_program_.id(), "blurScale"), screen_width / screen_aspect * (1.0f / (tanf(this->fov_*0.5f)))));

    glm::vec2 inv_text_scale = { 1.0f / screen_aspect, 1.0f };
    glVerify(glUniform2fv(glGetUniformLocation(this->depth_blur_program_.id(), "invTexScale"), 1, glm::value_ptr(inv_text_scale)));
    glVerify(glUniform1f(glGetUniformLocation(this->depth_blur_program_.id(), "blurFalloff"), this->fluid_blur_));
    glVerify(glUniform1i(glGetUniformLocation(this->depth_blur_program_.id(), "depthTex"), 0));
    glVerify(glUniform1i(glGetUniformLocation(this->depth_blur_program_.id(), "thicknessTex"), 1));
    glVerify(glUniform1i(glGetUniformLocation(this->depth_blur_program_.id(), "debug"), this->draw_opaque_));

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //render to depth_FBO
    glVerify(this->renderToFullScreenQuad());
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    glActiveTexture(GL_TEXTURE0);
    glDisable(GL_TEXTURE_RECTANGLE);

    /////////////////////////////////////////////////////////////////////////////////////////////////
    //switch to msaa_FBO, composite with scene

    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, msaa_FBO));
    glVerify(glEnable(GL_DEPTH_TEST));
    glVerify(glDepthMask(GL_TRUE));
    glVerify(glDisable(GL_BLEND));
    glVerify(glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA));

    this->composite_program_.use();

    //calculate light projection matrix
    glm::mat4 light_proj_mat = glm::perspective(screen_based_render->lightFov(), 1.0f, 0.1f, 1000.0f);

    //calculate light model view matrix
    glm::vec3 light_pos = { screen_based_render->lightPos()[0], screen_based_render->lightPos()[1], screen_based_render->lightPos()[2] };
    glm::vec3 light_target = { screen_based_render->lightTarget()[0], screen_based_render->lightTarget()[1], screen_based_render->lightTarget()[2] };
    glm::vec3 light_dir = glm::normalize(light_target - light_pos);
    glm::vec3 light_up = { 0.0f, 1.0f, 0.0f };


    glm::mat4 light_model_view_mat = glm::lookAt(light_pos, light_target, light_up);

    //calculate light transform matrix
    glm::mat4 light_transform_mat = light_proj_mat*light_model_view_mat;

    glm::vec4 fluid_color = {
                               this->fluid_color_.redChannel(),
                               this->fluid_color_.greenChannel(),
                               this->fluid_color_.blueChannel(),
                               this->fluid_color_.alphaChannel()
                             };

    glVerify(glUniform2fv(glGetUniformLocation(this->composite_program_.id(), "invTexScale"), 1, glm::value_ptr(glm::vec2(1.0f / screen_width, 1.0f / screen_height))));
    glVerify(glUniform2fv(glGetUniformLocation(this->composite_program_.id(), "clipPosToEye"), 1, glm::value_ptr(glm::vec2(tanf(this->fov_*0.5f)*screen_aspect, tanf(this->fov_*0.5f)))));

    glVerify(glUniform4fv(glGetUniformLocation(this->composite_program_.id(), "color"), 1, glm::value_ptr(fluid_color)));
    glVerify(glUniform1f(glGetUniformLocation(this->composite_program_.id(), "ior"), this->fluid_ior_));
    glVerify(glUniform1f(glGetUniformLocation(this->composite_program_.id(), "spotMin"), screen_based_render->lightSpotMin()));
    glVerify(glUniform1f(glGetUniformLocation(this->composite_program_.id(), "spotMax"), screen_based_render->lightSpotMax()));
    glVerify(glUniform1i(glGetUniformLocation(this->composite_program_.id(), "debug"), this->draw_opaque_));

    glVerify(glUniform3fv(glGetUniformLocation(this->composite_program_.id(), "lightPos"), 1, glm::value_ptr(light_pos)));
    glVerify(glUniform3fv(glGetUniformLocation(this->composite_program_.id(), "lightDir"), 1, glm::value_ptr(-light_dir))); // note: "-"
    glVerify(glUniformMatrix4fv(glGetUniformLocation(this->composite_program_.id(), "lightTransform"), 1, false, glm::value_ptr(light_transform_mat)));


    float shadow_taps[24] = {
                                -0.326212f, -0.40581f,  -0.840144f,  -0.07358f,
                                -0.695914f,  0.457137f, -0.203345f,   0.620716f,
                                 0.96234f,  -0.194983f,  0.473434f,  -0.480026f,
                                 0.519456f,  0.767022f,  0.185461f,  -0.893124f,
                                 0.507431f,  0.064425f,  0.89642f,    0.412458f,
                                -0.32194f,  -0.932615f,  -0.791559f, -0.59771f
                             };

    glVerify(glUniform2fv(glGetUniformLocation(this->composite_program_.id(), "shadowTaps"), 12, shadow_taps));

    // smoothed depth tex
    glVerify(glActiveTexture(GL_TEXTURE0));
    glVerify(glEnable(GL_TEXTURE_2D));
    glVerify(glBindTexture(GL_TEXTURE_2D, this->depth_smooth_TEX_));

    // shadow tex
    glVerify(glActiveTexture(GL_TEXTURE1));
    glVerify(glEnable(GL_TEXTURE_2D));
    glVerify(glBindTexture(GL_TEXTURE_2D, shadow_map_TEX));

    // thickness tex
    glVerify(glActiveTexture(GL_TEXTURE2));
    glVerify(glEnable(GL_TEXTURE_2D));
    glVerify(glBindTexture(GL_TEXTURE_2D, this->thickness_TEX_));

    // scene tex
    glVerify(glActiveTexture(GL_TEXTURE3));
    glVerify(glEnable(GL_TEXTURE_2D));
    glVerify(glBindTexture(GL_TEXTURE_2D, this->scene_TEX_));

    /*
    // reflection tex
    glVerify(glActiveTexture(GL_TEXTURE5));
    glVerify(glEnable(GL_TEXTURE_2D));
    glVerify(glBindTexture(GL_TEXTURE_2D, reflectMap->texture));
    */

    glVerify(glUniform1i(glGetUniformLocation(this->composite_program_.id(), "tex"), 0));
    glVerify(glUniform1i(glGetUniformLocation(this->composite_program_.id(), "shadowTex"), 1));
    glVerify(glUniform1i(glGetUniformLocation(this->composite_program_.id(), "thicknessTex"), 2));
    glVerify(glUniform1i(glGetUniformLocation(this->composite_program_.id(), "sceneTex"), 3));
    glVerify(glUniform1i(glGetUniformLocation(this->composite_program_.id(), "reflectTex"), 5));

    /////////////////////////////////////////////////////////////////////////////////////////////////

    /***************************************************************************************/
    //render to msaa_FBO
    glVerify(this->renderToFullScreenQuad());
    /***************************************************************************************/

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

void FluidRender::renderDiffuseParticle(ScreenBasedRenderManager * screen_based_render, GLuint shadow_map_TEX, bool front)
{
    glEnable(GL_POINT_SPRITE);
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    glDepthMask(GL_FALSE);
    glEnable(GL_DEPTH_TEST);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    glDisable(GL_CULL_FACE);

    this->diffuse_program_.use();

    float thickness_scale = 1;
    unsigned int screen_width  = screen_based_render->glutWindow()->width();
    unsigned int screen_height = screen_based_render->glutWindow()->height();
    float        screen_aspect = static_cast<float>(screen_width) / screen_height;
    glm::vec2    inv_viewport = { 1.0f / screen_width, 1.0f / screen_height };

    glm::vec4    diffuse_color = {
                                  this->diffuse_color_.redChannel(),
                                  this->diffuse_color_.greenChannel(),
                                  this->diffuse_color_.blueChannel(),
                                  this->diffuse_color_.alphaChannel()
                                 };

    glUniform1f(glGetUniformLocation( this->diffuse_program_.id(), "motionBlurScale"), this->diffuse_motion_scale_);
    glUniform1f(glGetUniformLocation( this->diffuse_program_.id(), "diffusion"), 1.0f);
    glUniform1f(glGetUniformLocation( this->diffuse_program_.id(), "pointScale"), this->radius_*this->diffuse_scale_);
    glUniform1f(glGetUniformLocation( this->diffuse_program_.id(), "pointRadius"), screen_width / thickness_scale / (2.0f*screen_aspect*tanf(this->fov_*0.5f)));
    glUniform2fv(glGetUniformLocation(this->diffuse_program_.id(), "invViewport"), 1, glm::value_ptr(inv_viewport));
    glUniform4fv(glGetUniformLocation(this->diffuse_program_.id(), "color"), 1, glm::value_ptr(diffuse_color));
    glUniform1i(glGetUniformLocation( this->diffuse_program_.id(), "tex"), 0);
    glUniform1f(glGetUniformLocation( this->diffuse_program_.id(), "inscatterCoefficient"), this->diffuse_inscatter_);
    glUniform1f(glGetUniformLocation( this->diffuse_program_.id(), "outscatterCoefficient"), this->diffuse_outscatter_);

    //calculate light projection matrix
    glm::mat4 light_proj_mat = glm::perspective(screen_based_render->lightFov(), 1.0f, 0.1f, 1000.0f);

    //calculate light model view matrix
    glm::vec3 light_pos = { screen_based_render->lightPos()[0], screen_based_render->lightPos()[1], screen_based_render->lightPos()[2] };
    glm::vec3 light_target = { screen_based_render->lightTarget()[0], screen_based_render->lightTarget()[1], screen_based_render->lightTarget()[2] };
    glm::vec3 light_dir = glm::normalize(light_target - light_pos);
    glm::vec3 light_up = { 0.0f, 1.0f, 0.0f };


    glm::mat4 light_model_view_mat = glm::lookAt(light_pos, light_target, light_up);

    //calculate light transform matrix
    glm::mat4 light_transform_mat = light_proj_mat*light_model_view_mat;

    glVerify(glUniformMatrix4fv(glGetUniformLocation(this->diffuse_program_.id(), "lightTransform"), 1, false, glm::value_ptr(light_transform_mat)));
    glVerify(glUniform3fv(glGetUniformLocation(this->diffuse_program_.id(), "lightPos"), 1, glm::value_ptr(light_pos)));
    glVerify(glUniform3fv(glGetUniformLocation(this->diffuse_program_.id(), "lightDir"), 1, glm::value_ptr(light_dir)));

    glUniform1f(glGetUniformLocation(this->diffuse_program_.id(), "spotMin"), screen_based_render->lightSpotMin());
    glUniform1f(glGetUniformLocation(this->diffuse_program_.id(), "spotMax"), screen_based_render->lightSpotMax());

    float shadow_taps[24] = {
                                -0.326212f, -0.40581f,  -0.840144f,  -0.07358f,
                                -0.695914f,  0.457137f, -0.203345f,   0.620716f,
                                 0.96234f,  -0.194983f,  0.473434f,  -0.480026f,
                                 0.519456f,  0.767022f,  0.185461f,  -0.893124f,
                                 0.507431f,  0.064425f,  0.89642f,    0.412458f,
                                -0.32194f,  -0.932615f,  -0.791559f, -0.59771f
                             };

    glVerify(glUniform2fv(glGetUniformLocation(this->diffuse_program_.id(), "shadowTaps"), 12, shadow_taps));

    glVerify(glUniform1i(glGetUniformLocation(this->diffuse_program_.id(), "depthTex"), 0));
    glVerify(glUniform1i(glGetUniformLocation(this->diffuse_program_.id(), "shadowTex"), 1));
    glVerify(glUniform1i(glGetUniformLocation(this->diffuse_program_.id(), "noiseTex"), 2));
    
    
    glVerify(glUniform1i(glGetUniformLocation(this->diffuse_program_.id(), "front"), front));
    glVerify(glUniform1i(glGetUniformLocation(this->diffuse_program_.id(), "shadow"), this->diffuse_shadow_));

    //depth smooth tex
    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, this->depth_smooth_TEX_);

    //shadow tex
    glActiveTexture(GL_TEXTURE1);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, shadow_map_TEX);
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE));

   
    glClientActiveTexture(GL_TEXTURE1);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glVerify(glBindBuffer(GL_ARRAY_BUFFER, this->diffuse_particle_buffer_.diffuse_velocity_VBO_));
    glTexCoordPointer(4, GL_FLOAT, sizeof(float) * 4, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, this->diffuse_particle_buffer_.diffuse_position_VBO_);
    glVertexPointer(4, GL_FLOAT, sizeof(float) * 4, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->diffuse_particle_buffer_.diffuse_indices_EBO_);

    //need further consideration
    glDrawElements(GL_POINTS, this->diffuse_particle_buffer_.diffuse_particle_num, GL_UNSIGNED_INT, 0);

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

void FluidRender::renderPoint(ScreenBasedRenderManager * screen_based_render)
{
    //redefine diff_program, needs further consideration
    ShaderProgram point_program;
    point_program.createFromCStyleString(vertex_point_shader, fragment_point_shader);

    glEnable(GL_POINT_SPRITE);
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    //glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    //needs further consideration
    bool show_density = false;
    bool use_shadow_map = true;

    int mode = 0;
    if (show_density)
        mode = 1;
    if (use_shadow_map == false)
        mode = 2;

    point_program.use();

    unsigned int screen_width = screen_based_render->glutWindow()->width();
    unsigned int screen_height = screen_based_render->glutWindow()->height();
    float        screen_aspect = static_cast<float>(screen_width) / screen_height;
    
    glVerify(glUniform1f(glGetUniformLocation(point_program.id(), "pointRadius"), this->radius_));
    glVerify(glUniform1f(glGetUniformLocation(point_program.id(), "pointScale"), screen_width / screen_aspect * (1.0f / (tanf(this->fov_*0.5f)))));
    glVerify(glUniform1f(glGetUniformLocation(point_program.id(), "spotMin"), screen_based_render->lightSpotMin()));
    glVerify(glUniform1f(glGetUniformLocation(point_program.id(), "spotMax"), screen_based_render->lightSpotMax()));
    glVerify(glUniform1i(glGetUniformLocation(point_program.id(), "mode"), mode));

    float colors[32] = {
                          0.0f,   0.5f,   1.0f,   1.0f,
                          0.797f, 0.354f, 0.000f, 1.0f,
                          0.092f, 0.465f, 0.820f, 1.0f,
                          0.000f, 0.349f, 0.173f, 1.0f,
                          0.875f, 0.782f, 0.051f, 1.0f,
                          0.000f, 0.170f, 0.453f, 1.0f,
                          0.673f, 0.111f, 0.000f, 1.0f,
                          0.612f, 0.194f, 0.394f, 1.0f,
                        };

    glVerify(glUniform4fv(glGetUniformLocation(point_program.id(), "colors"), 8, colors));

    // set shadow parameters
    screen_based_render->applyShadowMapWithSpecifiedShaderProgram(point_program);

    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, this->fluid_particle_buffer_.position_VBO_);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    int d = glGetAttribLocation(point_program.id(), "density");
    int p = glGetAttribLocation(point_program.id(), "phase");

    // densities
    if (d != -1)
    {
        glVerify(glEnableVertexAttribArray(d));
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, this->fluid_particle_buffer_.density_VBO_));
        glVerify(glVertexAttribPointer(d, 1, GL_FLOAT, GL_FALSE, 0, 0));	
    }

    // phases
    if (p != -1)
    {
        glVerify(glEnableVertexAttribArray(p));
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, this->fluid_particle_buffer_.density_VBO_));
        glVerify(glVertexAttribIPointer(p, 1, GL_INT, 0, 0));			
    }

    glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->fluid_particle_buffer_.indices_EBO_));

    glVerify(glDrawElements(GL_POINTS, this->fluid_particle_buffer_.fluid_particle_num, GL_UNSIGNED_INT, 0));

    
    point_program.unUse();

    glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));
    glVerify(glDisableClientState(GL_VERTEX_ARRAY));

    if (d != -1)
        glVerify(glDisableVertexAttribArray(d));
    if (p != -1)
        glVerify(glDisableVertexAttribArray(p));

    glDisable(GL_POINT_SPRITE);
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
}

void FluidRender::render(ScreenBasedRenderManager * screen_based_render, GLuint shadow_map_TEX)
{
    if (this->draw_diffuse_particle_)
        this->renderDiffuseParticle(screen_based_render, shadow_map_TEX, false);

    if (this->draw_fluid_particle_)
    {
        // render fluid surface
        this->renderFluidParticle(screen_based_render, shadow_map_TEX);

        // second pass of diffuse particles for particles in front of fluid surface
        if (this->draw_diffuse_particle_)
            this->renderDiffuseParticle(screen_based_render, shadow_map_TEX, true);
    }
    else
    {
        // draw all particles as spheres
        if (this->draw_point_)
        {
            this->renderPoint(screen_based_render);
        }
    }
}

GLuint FluidRender::fluidPositionVBO() const
{
    return this->fluid_particle_buffer_.position_VBO_;
}

void FluidRender::setDrawFluidParticle(bool draw_fluid_particle)
{
    this->draw_fluid_particle_ = draw_fluid_particle;
}

void FluidRender::setDrawDiffuseParticle(bool draw_diffuse_particle)
{
    this->draw_diffuse_particle_ = draw_diffuse_particle;
}

void FluidRender::setDrawOpaque(bool draw_opaque)
{
    this->draw_opaque_ = draw_opaque;
}

void FluidRender::setDrawPoint(bool draw_point)
{
    this->draw_point_ = draw_point;
}

void FluidRender::setFluidRadius(float fluid_radius)
{
    this->radius_ = fluid_radius;
}

void FluidRender::setFluidBlur(float fluid_blur)
{
    this->fluid_blur_ = fluid_blur;
}

void FluidRender::setFluidIor(float fluid_ior)
{
    this->fluid_ior_ = fluid_ior;
}

void FluidRender::setFluidColor(Color<float> fluid_color)
{
    this->fluid_color_ = fluid_color;
}

void FluidRender::setDiffuseColor(Color<float> diffuse_color)
{
    this->diffuse_color_ = diffuse_color;
}

void FluidRender::setDiffuseScale(float diffuse_scale)
{
    this->diffuse_scale_ = diffuse_scale;
}

void FluidRender::setDiffuseMotionScale(float diffuse_motion_scale)
{
    this->diffuse_motion_scale_ = diffuse_motion_scale;
}

void FluidRender::setDiffuseInscatter(float diffuse_inscatter)
{
    this->diffuse_inscatter_ = diffuse_inscatter;
}

void FluidRender::setDiffuseOutscatter(float diffuse_outscatter)
{
    this->diffuse_outscatter_ = diffuse_outscatter;
}

void FluidRender::setDiffuseShadow(bool diffuse_shadow)
{
    this->diffuse_shadow_ = diffuse_shadow;
}

void FluidRender::renderToFullScreenQuad()
{
    glColor3f(1.0f, 1.0f, 1.0f);

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

}// end of namespace Physika