/*
 * @file fluid_render_util.cpp
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

#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"

#include "fluid_render_util.h"

namespace Physika{

FluidRenderUtil::FluidRenderUtil(unsigned int fluid_particle_num, unsigned int diffuse_particle_num)
{
    this->initFluidParticleBuffer(fluid_particle_num);
    this->initDiffuseParticleBuffer(diffuse_particle_num);

    this->initFluidPointRenderVAO();
}

FluidRenderUtil::~FluidRenderUtil()
{
    this->destroyFluidParticleBuffer();
    this->destroyDiffusePartcileBuffer();

    this->destroyFluidPointRenderVAO();
}

void FluidRenderUtil::initFluidParticleBuffer(unsigned int fluid_particle_num)
{
    fluid_particle_buffer_.fluid_particle_num = fluid_particle_num;

    // create position_VBO
    glVerify(glGenBuffers(1, &fluid_particle_buffer_.position_VBO_));
    glVerify(glBindBuffer(GL_ARRAY_BUFFER, fluid_particle_buffer_.position_VBO_));
    glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * fluid_particle_num, 0, GL_DYNAMIC_DRAW));
    glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));

    // create density_VBO
    glVerify(glGenBuffers(1, &fluid_particle_buffer_.density_VBO_));
    glVerify(glBindBuffer(GL_ARRAY_BUFFER, fluid_particle_buffer_.density_VBO_));
    glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float)*fluid_particle_num, 0, GL_DYNAMIC_DRAW));
    glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));

    // create anisotropy_VBO
    for (int i = 0; i < 3; ++i)
    {
        glVerify(glGenBuffers(1, &fluid_particle_buffer_.anisotropy_VBO_[i]));
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, fluid_particle_buffer_.anisotropy_VBO_[i]));
        glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * fluid_particle_num, 0, GL_DYNAMIC_DRAW));
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));
    }

    // create indices_EBO
    glVerify(glGenBuffers(1, &fluid_particle_buffer_.indices_EBO_));
    glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, fluid_particle_buffer_.indices_EBO_));
    glVerify(glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int)*fluid_particle_num, 0, GL_DYNAMIC_DRAW));
    glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
}

void FluidRenderUtil::initDiffuseParticleBuffer(unsigned int diffuse_partcle_num)
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

void FluidRenderUtil::initFluidPointRenderVAO()
{
    //generate & bind VAO 
    glGenVertexArrays(1, &fluid_point_render_VAO_);
    glBindVertexArray(fluid_point_render_VAO_);

    //pos
    glBindBuffer(GL_ARRAY_BUFFER, this->fluid_particle_buffer_.position_VBO_);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //density
    glVerify(glBindBuffer(GL_ARRAY_BUFFER, this->fluid_particle_buffer_.density_VBO_));
        glVerify(glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, 0, nullptr));   //use vertex attribute index: 5
        glEnableVertexAttribArray(5);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //phase?? why use density ?? need further consideration
    glVerify(glBindBuffer(GL_ARRAY_BUFFER, this->fluid_particle_buffer_.density_VBO_));
        glVerify(glVertexAttribPointer(6, 1, GL_INT, GL_FALSE, 0, nullptr));
        glEnableVertexAttribArray(6);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //EBO
    glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->fluid_particle_buffer_.indices_EBO_));

    //unbind VAO
    glBindVertexArray(0);

    //unbind EBO
    glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
}

void FluidRenderUtil::destroyFluidParticleBuffer()
{
    glDeleteBuffers(1, &fluid_particle_buffer_.position_VBO_);
    glDeleteBuffers(1, &fluid_particle_buffer_.density_VBO_);
    glDeleteBuffers(3, fluid_particle_buffer_.anisotropy_VBO_);
    glDeleteBuffers(1, &fluid_particle_buffer_.indices_EBO_);
}

void FluidRenderUtil::destroyDiffusePartcileBuffer()
{
    if (diffuse_particle_buffer_.diffuse_particle_num > 0)
    {
        glDeleteBuffers(1, &diffuse_particle_buffer_.diffuse_position_VBO_);
        glDeleteBuffers(1, &diffuse_particle_buffer_.diffuse_velocity_VBO_);
        glDeleteBuffers(1, &diffuse_particle_buffer_.diffuse_indices_EBO_);
    }
}

void FluidRenderUtil::destroyFluidPointRenderVAO()
{
    glDeleteVertexArrays(1, &fluid_point_render_VAO_);
}


void FluidRenderUtil::updateFluidParticleBuffer(float * position_buffer, 
                                                float * density_buffer, 
                                                float * anisotropy_buffer_0, 
                                                float * anisotropy_buffer_1, 
                                                float * anisotropy_buffer_2, 
                                                unsigned int * indices_buffer, 
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
    glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));

    //update density_VBO
    if (density_buffer)
    {
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, fluid_particle_buffer_.density_VBO_));
        glVerify(glBufferSubData(GL_ARRAY_BUFFER, 0, density_buffer_size, density_buffer));
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));
    }

    //update anisotropy_buffer_VBO
    if (anisotropy_buffer_0)
    {
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, fluid_particle_buffer_.anisotropy_VBO_[0]));
        glVerify(glBufferSubData(GL_ARRAY_BUFFER, 0, anisotropy_buffer_size, anisotropy_buffer_0));
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));
    }

    if (anisotropy_buffer_1)
    {
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, fluid_particle_buffer_.anisotropy_VBO_[1]));
        glVerify(glBufferSubData(GL_ARRAY_BUFFER, 0, anisotropy_buffer_size, anisotropy_buffer_1));
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));
    }

    if (anisotropy_buffer_2)
    {
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, fluid_particle_buffer_.anisotropy_VBO_[2]));
        glVerify(glBufferSubData(GL_ARRAY_BUFFER, 0, anisotropy_buffer_size, anisotropy_buffer_2));
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));
    }

    
    //update indices_EBO
    if (indices_buffer)
    {
        glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, fluid_particle_buffer_.indices_EBO_));
        glVerify(glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, incides_buffer_size, indices_buffer));
        glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
    }
}

void FluidRenderUtil::updateDiffuseParticleBuffer(float * diffuse_position_buffer,
                                                  float * diffuse_velocity_buffer,
                                                  unsigned int * diffuse_indices_buffer)
{
    if (diffuse_particle_buffer_.diffuse_particle_num)
    {
        unsigned int position_buffer_size = diffuse_particle_buffer_.diffuse_particle_num * 4 * sizeof(float); //4*n*sizeof(float)
        unsigned int velocity_buffer_size = position_buffer_size; //4*n*sizeof(float)

        unsigned int indices_buffer_size = diffuse_particle_buffer_.diffuse_particle_num * sizeof(int);

        //update position_VBO
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, diffuse_particle_buffer_.diffuse_position_VBO_));
        glVerify(glBufferSubData(GL_ARRAY_BUFFER, 0, position_buffer_size, diffuse_position_buffer));
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));
        
        //update velocity_VBO
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, diffuse_particle_buffer_.diffuse_velocity_VBO_));
        glVerify(glBufferSubData(GL_ARRAY_BUFFER, 0, velocity_buffer_size, diffuse_velocity_buffer));
        glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));

        //update indices_EBO
        glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, diffuse_particle_buffer_.diffuse_indices_EBO_));
        glVerify(glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, indices_buffer_size, diffuse_indices_buffer));
        glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
    }
}

void FluidRenderUtil::drawByPoint()
{
    glBindVertexArray(fluid_point_render_VAO_);
    glVerify(glDrawElements(GL_POINTS, this->fluid_particle_buffer_.fluid_particle_num, GL_UNSIGNED_INT, 0));
    glBindVertexArray(0);
    
}
    
}//end of namespace Physika