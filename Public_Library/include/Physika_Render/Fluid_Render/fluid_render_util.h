/*
 * @file fluid_render_util.h 
 * @Brief class FluidRenderUtil
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

namespace Physika{

struct FluidParticleBuffer
{
    unsigned int position_VBO_ = 0; //4*n
    unsigned int density_VBO_ = 0;  //n
    unsigned int anisotropy_VBO_[3] = { 0, 0, 0 }; //{4*n, 4*n, 4*n}
    unsigned int indices_EBO_ = 0;

    unsigned int fluid_particle_num = 0;
};

struct DiffuseParticleBuffer
{
    unsigned int diffuse_position_VBO_ = 0; //4*n
    unsigned int diffuse_velocity_VBO_ = 0; //4*n
    unsigned int diffuse_indices_EBO_ = 0;

    unsigned int diffuse_particle_num = 0;
};

class FluidRenderUtil
{
public:

    FluidRenderUtil(unsigned int fluid_particle_num, unsigned int diffuse_particle_num);
    ~FluidRenderUtil();

    FluidRenderUtil(const FluidRenderUtil &) = delete;
    FluidRenderUtil & operator = (const FluidRenderUtil &) = delete;

    //need further consideration
    void updateFluidParticleBuffer(float * position_buffer,
                                   float * density_buffer,
                                   float * anisotropy_buffer_0,
                                   float * anisotropy_buffer_1,
                                   float * anisotropy_buffer_2,
                                   unsigned int * indices_buffer,
                                   unsigned int indices_num);

    void updateDiffuseParticleBuffer(float * diffuse_position_buffer,
                                     float * diffuse_velocity_buffer,
                                     unsigned int  * diffuse_indices_buffer);

    void drawByPoint();

    //need further consideration
    unsigned int fluidParticleNum() const { return this->fluid_particle_buffer_.fluid_particle_num; }
    unsigned int fluidParticlePositionVBO() const { return this->fluid_particle_buffer_.position_VBO_; }
    unsigned int fluidParticleAnisotropyVBO(int index) const { return this->fluid_particle_buffer_.anisotropy_VBO_[index]; }

    //need further consideration
    unsigned int diffuseParticleNum() const { return this->diffuse_particle_buffer_.diffuse_particle_num; }
    unsigned int diffuseParticlePositionVBO() const { return this->diffuse_particle_buffer_.diffuse_position_VBO_; }
    unsigned int diffuseParticleVelocityVBO() const { return this->diffuse_particle_buffer_.diffuse_velocity_VBO_; }
    unsigned int diffuseParticleEBO() const { return this->diffuse_particle_buffer_.diffuse_indices_EBO_; }

private:
    void initFluidParticleBuffer(unsigned int fluid_particle_num);
    void initDiffuseParticleBuffer(unsigned int diffuse_partcle_num);

    void initFluidPointRenderVAO();
    
    void destroyFluidParticleBuffer();
    void destroyDiffusePartcileBuffer();

    void destroyFluidPointRenderVAO();

private:
    FluidParticleBuffer fluid_particle_buffer_;
    DiffuseParticleBuffer diffuse_particle_buffer_;

    unsigned int fluid_point_render_VAO_;
};

}//end of namespace Physika