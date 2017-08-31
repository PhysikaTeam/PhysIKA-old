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
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Vectors/vector_4d.h"
#include "Physika_GUI/Glut_Window/glut_window.h"
#include "Physika_Render/Color/color.h"
#include "Physika_Render/OpenGL_Shaders/shaders.h"
#include "Physika_Render/Render_Base/render_base.h"
#include "Physika_Render/Surface_Mesh_Render/surface_mesh_render.h"
#include "Physika_Render/Screen_Based_Render_Manager/screen_based_render_manager.h"

namespace Physika {

/*
ScreenBasedRenderManager::ScreenBasedRenderManager()
{
    this->initMsaaFrameBuffer();
    this->shape_program_.createFromCStyleString(vertex_shader, fragment_shader);
}


ScreenBasedRenderManager::ScreenBasedRenderManager(unsigned int screen_width, unsigned int screen_height)
    :screen_width_(screen_width), screen_height_(screen_height)
{
    this->initMsaaFrameBuffer();
    this->shape_program_.createFromCStyleString(vertex_shader, fragment_shader);
}
*/

ScreenBasedRenderManager::ScreenBasedRenderManager(GlutWindow * glut_window)
    :glut_window_(glut_window), screen_width_(glut_window->width()), screen_height_(glut_window->height())
{
    this->initMsaaFrameBuffer();
    this->shape_program_.createFromCStyleString(vertex_shader, fragment_shader);
}

ScreenBasedRenderManager::~ScreenBasedRenderManager()
{
    this->destroyMsaaFrameBuffer();
}

void ScreenBasedRenderManager::initMsaaFrameBuffer()
{
    
    //switch to default render frame buffer, need further consideration
    //glVerify(glBindFramebuffer(GL_FRAMEBUFFER, 0));

    //determine msaa_samples
    int samples;
    glGetIntegerv(GL_MAX_SAMPLES_EXT, &samples);
    samples = std::min({ samples, this->msaa_samples_, 4 }); // clamp samples to 4 to avoid problems with point sprite scaling

    //create msaa_FBO
    glVerify(glGenFramebuffers(1, &this->msaa_FBO_));
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, this->msaa_FBO_));

    //create msaa_color_RBO & bind it to msaa_FBO
    glVerify(glGenRenderbuffers(1, &this->msaa_color_RBO_));
    glVerify(glBindRenderbuffer(GL_RENDERBUFFER, this->msaa_color_RBO_));
    glVerify(glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_RGBA8, this->screen_width_, this->screen_height_));
    glVerify(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, this->msaa_color_RBO_));

    //create msaa_depth_RBO & bind it to msaa_FBO
    glVerify(glGenRenderbuffers(1, &this->msaa_depth_RBO_));
    glVerify(glBindRenderbuffer(GL_RENDERBUFFER, this->msaa_depth_RBO_));
    glVerify(glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_DEPTH_COMPONENT, this->screen_width_, this->screen_height_));
    glVerify(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, this->msaa_depth_RBO_));

    //check frame buffer status
    glVerify(glCheckFramebufferStatus(GL_FRAMEBUFFER));

    glEnable(GL_MULTISAMPLE);
}

void ScreenBasedRenderManager::destroyMsaaFrameBuffer()
{
    glVerify(glDeleteFramebuffers(1, &this->msaa_FBO_));
    glVerify(glDeleteRenderbuffers(1, &this->msaa_color_RBO_));
    glVerify(glDeleteRenderbuffers(1, &this->msaa_depth_RBO_));
}

void ScreenBasedRenderManager::render()
{
    //begin frame, background color needs further consideration
    this->beginFrame(Color<float>::Black());

    //create shadow map
    this->createShadowMap();

    //create lighting map, i.e. draw all shapes with light
    this->createCameraMap();

    //render fluid
    if(this->fluid_render_)
        this->fluid_render_->render(this, this->shadow_map_.shadowTexId());

    //end frame
    this->endFrame();

}

const GlutWindow * ScreenBasedRenderManager::glutWindow() const
{
    return this->glut_window_;
}

const Vector<float, 3> & ScreenBasedRenderManager::lightPos() const
{
    return this->light_pos_;
}

const Vector<float, 3> & ScreenBasedRenderManager::lightTarget() const
{
    return this->light_target_;
}

float ScreenBasedRenderManager::lightFov() const
{
    return this->light_fov_;
}

float ScreenBasedRenderManager::lightSpotMin() const
{
    return this->light_spot_min_;
}

float ScreenBasedRenderManager::lightSpotMax() const
{
    return this->light_spot_max_;
}

void ScreenBasedRenderManager::addRender(RenderBase * render)
{
    this->render_list_.push_back(render);
}

void ScreenBasedRenderManager::addPlane(const Vector<float, 4> & plane)
{
    this->plans_.push_back(plane);
}

void ScreenBasedRenderManager::setGlutWindow(GlutWindow * glut_window)
{
    this->glut_window_ = glut_window;
}

void ScreenBasedRenderManager::setFluidRender(FluidRender * fluid_render)
{
    this->fluid_render_ = fluid_render;
}

void ScreenBasedRenderManager::setLightPos(const Vector<float, 3> & light_pos)
{
    this->light_pos_ = light_pos;
}

void ScreenBasedRenderManager::setLightTarget(const Vector<float, 3> & light_target)
{
    this->light_target_ = light_target;
}

void ScreenBasedRenderManager::setLightFov(float light_fov)
{
    this->light_fov_ = light_fov;
}

void ScreenBasedRenderManager::setLightSpotMin(float light_spot_min)
{
    this->light_spot_min_ = light_spot_min;
}

void ScreenBasedRenderManager::setLightSpotMax(float light_spot_max)
{
    this->light_spot_max_ = light_spot_max;
}

void ScreenBasedRenderManager::setFogDistance(float fog_distance)
{
    this->fog_distance_ = fog_distance;
}

void ScreenBasedRenderManager::setShadowBias(float shadow_bias)
{
    this->shadow_bias_ = shadow_bias;
}

GLuint ScreenBasedRenderManager::msaaFBO() const
{
    return this->msaa_FBO_;
}

const ShadowMap & ScreenBasedRenderManager::shadowMap() const
{
    return this->shadow_map_;
}

void ScreenBasedRenderManager::beginFrame(const Color<float> & clear_color)
{
    glEnable(GL_DEPTH_TEST);
    //glEnable(GL_CULL_FACE);

    glDisable(GL_LIGHTING);
    glDisable(GL_BLEND);

    //set point size
    glPointSize(5.0f);

    //switch to msaa_FBO
    glVerify(glBindFramebuffer(GL_DRAW_FRAMEBUFFER_EXT, this->msaa_FBO_));

    //need further consideration
    glVerify(glClearColor(clear_color.redChannel(), clear_color.greenChannel(), clear_color.blueChannel(), 0.0f));
    
    glVerify(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
}

void ScreenBasedRenderManager::createShadowMap()
{
    //begin shadow map
    this->shadow_map_.beginShadowMap();

    //set light project matrix & model view matrix
    this->setLightProjAndModelViewMatrix();

    //draw scene shapes
    this->drawShapes();

    //end shadow map
    this->shadow_map_.endShadowMap();

}

void ScreenBasedRenderManager::createCameraMap()
{
    //begin lighting
    this->beginLighting();

    //apply shadow map
    this->applyShadowMap();

    //set camera projection and model view matrix
    this->setCameraProjAndModelViewMatrix();

    //draw planes
    this->drawPlanes();

    //draw shapes
    this->drawShapes();

    //end lighting
    this->endLighting();
}

void ScreenBasedRenderManager::endFrame()
{
    //specify msaa_FBO to read and default FBO to draw
    glVerify(glBindFramebuffer(GL_READ_FRAMEBUFFER_EXT, this->msaa_FBO_));
    glVerify(glBindFramebuffer(GL_DRAW_FRAMEBUFFER_EXT, 0));

    //blit the msaa_FBO to the window(default FBO), i.e. render to the current window
    glVerify(glBlitFramebuffer(0, 0, this->screen_width_, this->screen_height_, 0, 0, this->screen_width_, this->screen_height_, GL_COLOR_BUFFER_BIT, GL_LINEAR));
    
    //render help to back buffer
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    glVerify(glClear(GL_DEPTH_BUFFER_BIT));

    //finish rendering
    glVerify(glFinish());

    //swap buffers
    glVerify(glutSwapBuffers());
}

void ScreenBasedRenderManager::beginLighting()
{
    //switch to msaa_FBO
    glVerify(glBindFramebuffer(GL_FRAMEBUFFER, this->msaa_FBO_));

    //set view port
    glVerify(glViewport(0, 0, this->screen_width_, this->screen_height_));

    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);
    //glEnable(GL_CULL_FACE);
    glDisable(GL_CULL_FACE);

    //use solid_diffuse_program
    this->shape_program_.use();
    
    glVerify(glUniform1i( glGetUniformLocation(this->shape_program_.id(), "grid"), 0));
    glVerify(glUniform1f( glGetUniformLocation(this->shape_program_.id(), "spotMin"), this->light_spot_min_));
    glVerify(glUniform1f( glGetUniformLocation(this->shape_program_.id(), "spotMax"), this->light_spot_max_));

    Color<float> background_color = glut_window_->backgroundColor<float>();
    glm::vec4 glm_fog_color = { background_color.redChannel(), background_color.greenChannel(), background_color.blueChannel(), this->fog_distance_ };
    glVerify(glUniform4fv(glGetUniformLocation(this->shape_program_.id(), "fogColor"), 1, glm::value_ptr(glm_fog_color)));

    glVerify(glUniformMatrix4fv(glGetUniformLocation(this->shape_program_.id(), "objectTransform"), 1, false, glm::value_ptr(glm::mat4(1.0))));
}

void ScreenBasedRenderManager::applyShadowMap()
{

    //calculate light projection matrix
    glm::mat4 light_proj_mat = glm::perspective(glm::radians(this->light_fov_), 1.0f, 1.0f, 1000.0f);

    //calculate light model view matrix
    glm::vec3 light_pos = { this->light_pos_[0], this->light_pos_[1], this->light_pos_[2] };
    glm::vec3 light_target = { this->light_target_[0], this->light_target_[1], this->light_target_[2] };
    glm::vec3 light_dir = glm::normalize(light_target - light_pos);
    glm::vec3 light_up = { 0.0f, 1.0f, 0.0f };


    glm::mat4 light_model_view_mat = glm::lookAt(light_pos, light_target, light_up);

    //calculate light transform matrix
    glm::mat4 light_transform_mat = light_proj_mat*light_model_view_mat;

    /*******************************************************************************************************************/
    std::cout <<"light_pos: "<< light_pos_ << std::endl;
    std::cout <<"light_target: "<< light_target_ << std::endl;
    std::cout <<"light_dir: "<< light_dir[0] << " " << light_dir[1] << " " << light_dir[2] << std::endl;
    std::cout << "------------------------------------" << std::endl;
    /*******************************************************************************************************************/

    glVerify(glUniformMatrix4fv(glGetUniformLocation(this->shape_program_.id(), "lightTransform"), 1, false, glm::value_ptr(light_transform_mat)));
    glVerify(glUniform3fv(      glGetUniformLocation(this->shape_program_.id(), "lightPos"), 1, glm::value_ptr(light_pos)));
    glVerify(glUniform3fv(      glGetUniformLocation(this->shape_program_.id(), "lightDir"), 1, glm::value_ptr(light_dir)));
    glVerify(glUniform1f(       glGetUniformLocation(this->shape_program_.id(), "bias"), this->shadow_bias_));


    float shadow_taps[24] = {
                                -0.326212f, -0.40581f,  -0.840144f,  -0.07358f,
                                -0.695914f,  0.457137f, -0.203345f,   0.620716f,
                                 0.96234f,  -0.194983f,  0.473434f,  -0.480026f,
                                 0.519456f,  0.767022f,  0.185461f,  -0.893124f,
                                 0.507431f,  0.064425f,  0.89642f,    0.412458f,
                                -0.32194f,  -0.932615f,  -0.791559f, -0.59771f
                             };

    glVerify(glUniform2fv(glGetUniformLocation(this->shape_program_.id(), "shadowTaps"), 12, shadow_taps));

    glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE0);

    //switch to shadow_TEX
    glVerify(glBindTexture(GL_TEXTURE_2D, this->shadow_map_.shadowTexId()));
}

void ScreenBasedRenderManager::applyShadowMapWithSpecifiedShaderProgram(ShaderProgram & shader_program)
{
    //calculate light projection matrix
    glm::mat4 light_proj_mat = glm::perspective(glm::radians(this->light_fov_), 1.0f, 1.0f, 1000.0f);

    //calculate light model view matrix
    glm::vec3 light_pos = { this->light_pos_[0], this->light_pos_[1], this->light_pos_[2] };
    glm::vec3 light_target = { this->light_target_[0], this->light_target_[1], this->light_target_[2] };
    glm::vec3 light_dir = glm::normalize(light_target - light_pos);
    glm::vec3 light_up = { 0.0f, 1.0f, 0.0f };


    glm::mat4 light_model_view_mat = glm::lookAt(light_pos, light_target, light_up);

    //calculate light transform matrix
    glm::mat4 light_transform_mat = light_proj_mat*light_model_view_mat;

    glVerify(glUniformMatrix4fv(glGetUniformLocation(shader_program.id(), "lightTransform"), 1, false, glm::value_ptr(light_transform_mat)));
    glVerify(glUniform3fv(glGetUniformLocation(shader_program.id(), "lightPos"), 1, glm::value_ptr(light_pos)));
    glVerify(glUniform3fv(glGetUniformLocation(shader_program.id(), "lightDir"), 1, glm::value_ptr(light_dir)));
    glVerify(glUniform1f(glGetUniformLocation(shader_program.id(), "bias"), this->shadow_bias_));


    float shadow_taps[24] = {
                              -0.326212f, -0.40581f,  -0.840144f,  -0.07358f,
                              -0.695914f,  0.457137f, -0.203345f,   0.620716f,
                               0.96234f,  -0.194983f,  0.473434f,  -0.480026f,
                               0.519456f,  0.767022f,  0.185461f,  -0.893124f,
                               0.507431f,  0.064425f,  0.89642f,    0.412458f,
                              -0.32194f,  -0.932615f,  -0.791559f, -0.59771f
                             };

    glVerify(glUniform2fv(glGetUniformLocation(shader_program.id(), "shadowTaps"), 12, shadow_taps));

    glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE0);

    //switch to shadow_TEX
    glVerify(glBindTexture(GL_TEXTURE_2D, this->shadow_map_.shadowTexId()));
}

void ScreenBasedRenderManager::endLighting()
{
    //unuse solid_diffuse_program
    this->shape_program_.unUse();

    //what the fuck????????????
    glActiveTexture(GL_TEXTURE1);
    glDisable(GL_TEXTURE_2D);

    glActiveTexture(GL_TEXTURE0);
}

void ScreenBasedRenderManager::drawShapes()
{
    glDisable(GL_CULL_FACE);
    for (RenderBase * render : this->render_list_)
        render->render();
}

void ScreenBasedRenderManager::drawPlanes()
{
    // diffuse 		
    glColor3f(0.9f, 0.9f, 0.9f);

    GLint bias = glGetUniformLocation(this->shape_program_.id(), "bias");
    glVerify(glUniform1f(bias, 0.0f));
    GLint grid = glGetUniformLocation(this->shape_program_.id(), "grid");
    glVerify(glUniform1i(grid, 1));
    GLint expand = glGetUniformLocation(this->shape_program_.id(), "expand");
    glVerify(glUniform1f(expand , 0.0f));

    for (int i = 0; i < this->plans_.size(); ++i)
    {
        this->drawPlane(this->plans_[i]);
    }

    glVerify(glUniform1i(grid, 0));
    glVerify(glUniform1f(bias, this->shadow_bias_));
}

void ScreenBasedRenderManager::drawPlane(const Vector<float, 4> & plane)
{
    Vector<float, 3> normal(plane[0], plane[1], plane[2]);
    Vector<float, 3> u, v;
    getBasisFromNormalVector(normal, u, v);

    Vector<float, 3> c = normal * -plane[3];

    /************************************************/
    //std::cout << "p: " << plane << std::endl;
    //std::cout << "u: " << u << std::endl;
    //std::cout << "v: " << v << std::endl;
    //std::cout << "c: " << c << std::endl;
    /************************************************/

    glBegin(GL_QUADS);

    float plane_size = 200.0f;

    // draw a grid of quads, otherwise z precision suffers
    for (int x = -3; x <= 3; ++x)
    {
        for (int y = -3; y <= 3; ++y)
        {
            Vector<float, 3> coff = c + u*float(x)*plane_size*2.0f + v*float(y)*plane_size*2.0f;
            //std::cout << "coff: " << coff << std::endl;

            glTexCoord2f(1.0f, 1.0f);
            glNormal3f(plane[0], plane[1], plane[2]);
            Vector<float, 3> v1 = coff + u*plane_size + v*plane_size;
            glVertex3f(v1[0], v1[1], v1[2]);

            //std::cout << "v1: " << v1 << std::endl;

            glTexCoord2f(0.0f, 1.0f);
            glNormal3f(plane[0], plane[1], plane[2]);
            Vector<float, 3> v2 = coff - u*plane_size + v*plane_size;
            glVertex3f(v2[0], v2[1], v2[2]);

            //std::cout << "v2: " << v2 << std::endl;

            glTexCoord2f(0.0f, 0.0f);
            glNormal3f(plane[0], plane[1], plane[2]);
            Vector<float, 3> v3 = coff - u*plane_size - v*plane_size;
            glVertex3f(v3[0], v3[1], v3[2]);

            //std::cout << "v3: " << v3 << std::endl;

            glTexCoord2f(1.0f, 0.0f);
            glNormal3f(plane[0], plane[1], plane[2]);
            Vector<float, 3> v4 = coff + u*plane_size - v*plane_size;
            glVertex3f(v4[0], v4[1], v4[2]);

            //std::cout << "v4: " << v4 << std::endl;
        }
    }

    glEnd();
}

void ScreenBasedRenderManager::getBasisFromNormalVector(const Vector<float, 3> & w, Vector<float, 3> & u, Vector<float, 3> & v)
{
    if (fabsf(w[0]) > fabsf(w[1]))
    {
        float inv_len = 1.0f / sqrtf(w[0] * w[0] + w[2] * w[2]);
        u = Vector<float, 3>(-w[2]*inv_len, 0.0f, w[0]*inv_len);
    }
    else
    {
        float inv_len = 1.0f / sqrtf(w[1] * w[1] + w[2] * w[2]);
        u = Vector<float, 3>(0.0f, w[2]*inv_len, -w[1]*inv_len);
    }

    v = w.cross(u);

}

void ScreenBasedRenderManager::setCameraProjAndModelViewMatrix()
{
    //calculate projection matrix
    glm::mat4 proj_mat = glm::perspective(glm::radians(this->glut_window_->cameraFOV()),
                                          this->glut_window_->cameraAspect(),
                                          this->glut_window_->cameraNearClip(),
                                          this->glut_window_->cameraFarClip());

    //calculate model view matrix
    const Vector<double, 3> & camera_pos = this->glut_window_->cameraPosition();
    const Vector<double, 3> & camera_focus_pos = this->glut_window_->cameraFocusPosition();
    const Vector<double, 3> & camera_up_dir = this->glut_window_->cameraUpDirection();

    /************************************************************************/
    std::cout << "camera_pos: " << camera_pos << std::endl;
    std::cout << "camera_focus_pos: " << camera_focus_pos << std::endl;
    std::cout << "camera_up_dir: " << camera_up_dir << std::endl;
    /************************************************************************/

    glm::vec3 glm_camera_pos = { camera_pos[0], camera_pos[1], camera_pos[2] };
    glm::vec3 glm_camera_focus_pos = { camera_focus_pos[0], camera_focus_pos[1], camera_focus_pos[2] };
    glm::vec3 glm_camera_up_dir = { camera_up_dir[0], camera_up_dir[1], camera_up_dir[2] };

    glm::mat4 model_view_mat = glm::lookAt(glm_camera_pos, glm_camera_focus_pos, glm_camera_up_dir);

    //set projection and model view matrix
    this->setProjAndModelViewMatrix(glm::value_ptr(proj_mat), glm::value_ptr(model_view_mat));
}

void ScreenBasedRenderManager::setLightProjAndModelViewMatrix()
{
    //calculate light projection matrix
    glm::mat4 light_proj_mat = glm::perspective(glm::radians(this->light_fov_), 1.0f, 1.0f, 1000.0f);

    //calculate light model view matrix
    glm::vec3 light_pos = { this->light_pos_[0], this->light_pos_[1], this->light_pos_[2] };
    glm::vec3 light_target = { this->light_target_[0], this->light_target_[1], this->light_target_[2] };
    glm::vec3 light_up = { 0.0f, 1.0f, 0.0f };

    glm::mat4 light_model_view_mat = glm::lookAt(light_pos, light_target, light_up);

    /************************************************************************************************************/
    std::cout << "light_pos: " << light_pos_ << std::endl;
    std::cout << "light_target: " << light_target_ << std::endl;
    std::cout << "light_fov: " << light_fov_ << std::endl;

    std::cout << "light_proj_mat: " << std::endl;
    for (int i = 0; i < 16; ++i)
        std::cout << glm::value_ptr(light_proj_mat)[i] << " ";
    std::cout << std::endl;

    std::cout << "light_model_view_mat: " << std::endl;
    for (int i = 0; i < 16; ++i)
        std::cout << glm::value_ptr(light_model_view_mat)[i] << " ";
    std::cout << std::endl;

    std::cout << "------------------------------------" << std::endl;
    /************************************************************************************************************/

    this->setProjAndModelViewMatrix(glm::value_ptr(light_proj_mat), glm::value_ptr(light_model_view_mat));
}

void ScreenBasedRenderManager::setProjAndModelViewMatrix(GLfloat * proj_mat, GLfloat * model_view_mat)
{
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(proj_mat);

    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(model_view_mat);
}


}//end of namespace Physika
