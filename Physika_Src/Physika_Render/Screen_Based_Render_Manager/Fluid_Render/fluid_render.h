/*
* @file fluid_render.h
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

#ifndef PHYSIKA_RENDER_SCREEN_BASED_RENDER_FLUID_RENDER_FLUID_RENDER_H
#define PHYSIKA_RENDER_SCREEN_BASED_RENDER_FLUID_RENDER_FLUID_RENDER_H

#include "Physika_Render/Color/color.h"
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include "Physika_Render/OpenGL_Shaders/shader_program.h"

namespace Physika{

//need further consideration

struct FluidParticleBuffer
{
    GLuint position_VBO_ = 0; //4*n
    GLuint density_VBO_ = 0;  //n
    GLuint anisotropy_VBO_[3] = { 0, 0, 0 }; //{4*n, 4*n, 4*n}
    GLuint indices_EBO_ = 0;

    /*
    GLfloat * position_buffer = nullptr; //4*n
    GLfloat * density_buffer = nullptr;  //n
    GLfloat * anisotropy_buffer[3] = { nullptr, nullptr, nullptr }; //{4*n, 4*n, 4*n}
    GLuint  * indices_buffer = nullptr;
    */

    unsigned int fluid_particle_num = 0;

};

struct DiffuseParticleBuffer
{
    GLuint diffuse_position_VBO_ = 0; //4*n
    GLuint diffuse_velocity_VBO_ = 0; //4*n
    GLuint diffuse_indices_EBO_ = 0;

    /*
    GLfloat * diffuse_position_buffer = nullptr; //4*n
    GLfloat * diffuse_velocity_buffer = nullptr; //4*n
    GLuint  * diffuse_indices_buffer = nullptr;  
    */

    unsigned int diffuse_particle_num = 0;
};

class ScreenBasedRenderManager;

class FluidRender
{
public:

    //need further consideration
    FluidRender(unsigned int fluid_particle_num, unsigned int diffuse_particle_num, unsigned int screen_width, unsigned int screen_height);

    ~FluidRender();

    FluidRender(const FluidRender &) = delete;
    FluidRender & operator = (const FluidRender &) = delete;

private:
    void initFluidParticleBuffer(unsigned int fluid_particle_num);
    void initDiffuseParticleBuffer(unsigned int diffuse_partcle_num);
    void initFrameBufferAndShaderProgram();

    void destroyFluidParticleBuffer();
    void destroyDiffusePartcileBuffer();
    void destroyFrameBufferAndShaderProgram();

public:

    void updateFluidParticleBuffer(GLfloat * position_buffer,
                                   GLfloat * density_buffer,
                                   GLfloat * anisotropy_buffer_0,
                                   GLfloat * anisotropy_buffer_1,
                                   GLfloat * anisotropy_buffer_2,
                                   GLuint  * indices_buffer,
                                   unsigned int indices_num);

    void updateDiffuseParticleBuffer(GLfloat * diffuse_position_buffer,
                                     GLfloat * diffuse_velocity_buffer,
                                     GLuint  * diffuse_indices_buffer);

    


    void renderFluidParticle(ScreenBasedRenderManager * screen_based_render, GLuint shadow_map_TEX);
    void renderDiffuseParticle(ScreenBasedRenderManager * screen_based_render, GLuint shadow_map_TEX, bool front);
    void renderPoint(ScreenBasedRenderManager * screen_based_render);

    void render(ScreenBasedRenderManager * screen_based_render, GLuint shadow_map_TEX);

    GLuint fluidPositionVBO() const;

    void setDrawFluidParticle(bool draw_fluid_particle);
    void setDrawDiffuseParticle(bool draw_diffuse_particle);
    void setDrawOpaque(bool draw_opaque);
    void setDrawPoint(bool draw_point);

    void setFluidRadius(float fluid_radius);
    void setFluidBlur(float fluid_blur);
    void setFluidIor(float fluid_ior);
    void setFluidColor(Color<float> fluid_color);

    void setDiffuseColor(Color<float> diffuse_color);
    void setDiffuseScale(float diffuse_scale);
    void setDiffuseMotionScale(float diffuse_motion_scale);
    void setDiffuseInscatter(float diffuse_inscatter);
    void setDiffuseOutscatter(float diffuse_outscatter);
    void setDiffuseShadow(bool diffuse_shadow);


private:

    void renderToFullScreenQuad();

private:

    FluidParticleBuffer fluid_particle_buffer_;
    DiffuseParticleBuffer diffuse_particle_buffer_;

    unsigned int screen_width_;
    unsigned int screen_height_;

    float radius_ = 0.01f;
    float fov_ = 3.14159f / 4.0f;

    float fluid_blur_ = 1.0f;
    float fluid_ior_ = 1.0f;
    Color<float> fluid_color_ = { 0.1f, 0.4f, 0.8f, 1.0f };

    Color<float> diffuse_color_ = { 1.0f, 1.0f, 1.0f, 1.0f };
    float        diffuse_scale_ = 0.5;
    float        diffuse_motion_scale_ = 1.0f;
    float        diffuse_inscatter_ = 0.8f;
    float        diffuse_outscatter_ = 0.53f;
    bool         diffuse_shadow_ = false;

    bool draw_fluid_particle_ = true;
    bool draw_diffuse_particle_ = false;
    bool draw_opaque_ = false; 
    bool draw_point_ = false;

    GLuint depth_FBO_ = 0;
    GLuint depth_TEX_ = 0;
    GLuint depth_smooth_TEX_ = 0;

    GLuint scene_FBO_ = 0;
    GLuint scene_TEX_ = 0;

    GLuint reflect_TEX_ = 0;

    GLuint thickness_FBO_ = 0;
    GLuint thickness_TEX_ = 0;

    ShaderProgram diffuse_program_;

    ShaderProgram point_thickness_program_;

    ShaderProgram ellipsoid_thickness_program_;
    ShaderProgram ellipsoid_depth_program_;

    ShaderProgram composite_program_;
    ShaderProgram depth_blur_program_;

};

} // end of namespace Physika

#endif // PHYSIKA_RENDER_SCREEN_BASED_RENDER_FLUID_RENDER_FLUID_RENDER_H
