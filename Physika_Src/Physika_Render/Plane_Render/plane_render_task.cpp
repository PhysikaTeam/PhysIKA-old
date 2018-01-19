/*
 * @file plane_render_task.cpp
 * @Basic render task of plane
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

#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"

#include "plane_shader_srcs.h"
#include "plane_render_task.h"


namespace Physika{

PlaneRenderTask::PlaneRenderTask()
{
    shader_.createFromCStyleString(plane_vertex_shader, plane_frag_shader);
}

PlaneRenderTask::~PlaneRenderTask()
{
    this->destoryAllVAOAndVBO();
}

void PlaneRenderTask::addPlane(const Vector4f & plane, float plane_size)
{
    this->planes_.push_back(plane);
    this->plane_sizes_.push_back(plane_size);

    this->addPlaneVAOAndVBO(plane, plane_size);
}

void PlaneRenderTask::enableRenderGrid()
{
    this->render_grid_ = true;
}

void PlaneRenderTask::disableRenderGrid()
{
    this->render_grid_ = false;
}

bool PlaneRenderTask::isRenderGrid() const
{
    return this->render_grid_;
}

void PlaneRenderTask::renderTaskImpl()
{
    if (this->plane_VAOs_.size() != this->plane_vert_nums_.size())
        throw PhysikaException("error: VAO size not match this->plane_vert_nums_.size()!");

    openGLSetCurBindShaderBool("render_grid", render_grid_);

    for(int i = 0; i < this->plane_VAOs_.size(); ++i)
    {
        glBindVertexArray(this->plane_VAOs_[i]);
        glDrawArrays(GL_TRIANGLES, 0, this->plane_vert_nums_[i]);
        glBindVertexArray(0);
    }
}

void PlaneRenderTask::addPlaneVAOAndVBO(const Vector4f & plane, float plane_size)
{
    Vector<float, 3> normal(plane[0], plane[1], plane[2]);
    Vector<float, 3> u, v;
    this->getBasisFromNormalVector(normal, u, v);

    Vector<float, 3> c = normal * -plane[3];

    //std::cout << "p: " << plane << std::endl;
    //std::cout << "u: " << u << std::endl;
    //std::cout << "v: " << v << std::endl;
    //std::cout << "c: " << c << std::endl;

    std::vector<glm::vec3> pos_vec;
    std::vector<glm::vec3> normal_vec;
    unsigned int plane_vert_num = 0;

    // draw a grid of quads, otherwise z precision suffers
    for (int x = -3; x <= 3; ++x)
    {
        for (int y = -3; y <= 3; ++y)
        {
            Vector<float, 3> coff = c + u*float(x)*plane_size*2.0f + v*float(y)*plane_size*2.0f;
            
            Vector<float, 3> v1 = coff + u*plane_size + v*plane_size;
            Vector<float, 3> v2 = coff - u*plane_size + v*plane_size;
            Vector<float, 3> v3 = coff - u*plane_size - v*plane_size;
            Vector<float, 3> v4 = coff + u*plane_size - v*plane_size;

            glm::vec3 glm_v1 = { v1[0], v1[1], v1[2] };
            glm::vec3 glm_v2 = { v2[0], v2[1], v2[2] };
            glm::vec3 glm_v3 = { v3[0], v3[1], v3[2] };
            glm::vec3 glm_v4 = { v4[0], v4[1], v4[2] };

            //first triangle
            pos_vec.push_back(glm_v1);
            pos_vec.push_back(glm_v2);
            pos_vec.push_back(glm_v3);

            //second triangle
            pos_vec.push_back(glm_v1);
            pos_vec.push_back(glm_v3);
            pos_vec.push_back(glm_v4);

            glm::vec3 glm_normal = { normal[0], normal[1], normal[2] };
            normal_vec.insert(normal_vec.end(), 6, glm_normal);
            plane_vert_num += 6;

            //std::cout << "coff: " << coff << std::endl;
            //std::cout << "v1: " << v1 << std::endl;
            //std::cout << "v2: " << v2 << std::endl;
            //std::cout << "v3: " << v3 << std::endl;
            //std::cout << "v4: " << v4 << std::endl;
        }
    }

    this->plane_vert_nums_.push_back(plane_vert_num);

    unsigned int VAO = 0;
    glVerify(glGenVertexArrays(1, &VAO));
    this->plane_VAOs_.push_back(VAO);

    glVerify(glBindVertexArray(VAO));
    
    unsigned int pos_VBO = 0;
    glGenBuffers(1, &pos_VBO);
    this->plane_VBOs_.push_back(pos_VBO);

    glBindBuffer(GL_ARRAY_BUFFER, pos_VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * pos_vec.size(), pos_vec.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    unsigned int normal_VBO = 0;
    glGenBuffers(1, &normal_VBO);
    this->plane_VBOs_.push_back(normal_VBO);

    glBindBuffer(GL_ARRAY_BUFFER, normal_VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * normal_vec.size(), normal_vec.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glVerify(glBindVertexArray(0));
}

void PlaneRenderTask::destoryAllVAOAndVBO()
{
    glVerify(glDeleteVertexArrays(this->plane_VAOs_.size(), this->plane_VAOs_.data()));
    glVerify(glDeleteBuffers(this->plane_VBOs_.size(), this->plane_VBOs_.data()));

    this->plane_VAOs_.clear();
    this->plane_VBOs_.clear();
    this->plane_vert_nums_.clear();
}

void PlaneRenderTask::getBasisFromNormalVector(const Vector<float, 3> & w, Vector<float, 3> & u, Vector<float, 3> & v)
{
    if (fabsf(w[0]) > fabsf(w[1]))
    {
        float inv_len = 1.0f / sqrtf(w[0] * w[0] + w[2] * w[2]);
        u = Vector<float, 3>(-w[2] * inv_len, 0.0f, w[0] * inv_len);
    }
    else
    {
        float inv_len = 1.0f / sqrtf(w[1] * w[1] + w[2] * w[2]);
        u = Vector<float, 3>(0.0f, w[2] * inv_len, -w[1] * inv_len);
    }
    v = w.cross(u);
}

}//end of namespace Physika
