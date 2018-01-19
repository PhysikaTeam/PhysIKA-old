/*
 * @file triangle_render_util.cpp 
 * @Basic class TriangleRenderUtil
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
#include "triangle_render_util.h"

namespace Physika{

TriangleRenderUtil::TriangleRenderUtil()
{
    this->initVAOAndVBOs();
}

TriangleRenderUtil::~TriangleRenderUtil()
{
    this->destoryVAOAndVBOs();
}

TriangleRenderUtil::TriangleRenderUtil(TriangleRenderUtil && rhs) noexcept
{
    this->triangle_num_ = rhs.triangle_num_;
    this->triangle_VBO_ = rhs.triangle_VBO_;
    this->normal_VBO_ = rhs.normal_VBO_;
    this->tex_coord_VBO_ = rhs.tex_coord_VBO_;
    this->triangle_VAO_ = rhs.triangle_VAO_;

    rhs.triangle_num_ = 0;
    rhs.triangle_VBO_ = 0;
    rhs.normal_VBO_ = 0;
    rhs.tex_coord_VBO_ = 0;
    rhs.triangle_VAO_ = 0;
}

TriangleRenderUtil & TriangleRenderUtil::operator = (TriangleRenderUtil && rhs) noexcept
{
    this->triangle_num_ = rhs.triangle_num_;
    this->triangle_VBO_ = rhs.triangle_VBO_;
    this->normal_VBO_ = rhs.normal_VBO_;
    this->tex_coord_VBO_ = rhs.tex_coord_VBO_;
    this->triangle_VAO_ = rhs.triangle_VAO_;

    rhs.triangle_num_ = 0;
    rhs.triangle_VBO_ = 0;
    rhs.normal_VBO_ = 0;
    rhs.tex_coord_VBO_ = 0;
    rhs.triangle_VAO_ = 0;

    return *this;
}

template <typename Scalar, int Dim>
void TriangleRenderUtil::setTriangles(const std::vector<Vector<Scalar, Dim>> & pos_vec, bool auto_compute_normal)
{
    if (pos_vec.size() % 3 != 0)
        throw PhysikaException("error: pos_vec.size() % 3 != 0!");

    std::vector<glm::vec3> glm_pos_vec(pos_vec.size());
    for (unsigned int i = 0; i < pos_vec.size(); ++i)
    {
        if (Dim == 2)
            glm_pos_vec[i] = { pos_vec[i][0], pos_vec[i][1], 0.0 };
        else
            glm_pos_vec[i] = { pos_vec[i][0], pos_vec[i][1], pos_vec[i][2] };
    }

    this->triangle_num_ = glm_pos_vec.size() / 3;
    this->updateTriangleVBOData(glm_pos_vec);

    if (auto_compute_normal == true)
        this->autoComputeNormal(glm_pos_vec);

}

template <typename Scalar, int Dim>
void TriangleRenderUtil::setTriangles(const std::vector<Vector<Scalar, Dim>> & pos_vec, std::vector<unsigned int> & indices, bool auto_compute_normal)
{
    if (pos_vec.size() < 3 || indices.size() < 3 || indices.size() % 3 != 0)
        throw PhysikaException("error: pos_vec.size() < 3 || indices.size() < 3 || indices.size() % 3 != 0!");

    std::vector<glm::vec3> glm_pos_vec(indices.size());
    for (unsigned int i = 0; i < indices.size(); ++i)
    {
        unsigned int index = indices[i];

        if (Dim == 2)
            glm_pos_vec[i] = { pos_vec[index][0], pos_vec[index][1], 0.0 };
        else
            glm_pos_vec[i] = { pos_vec[index][0], pos_vec[index][1], pos_vec[index][2] };
    }

    this->triangle_num_ = glm_pos_vec.size() / 3;
    this->updateTriangleVBOData(glm_pos_vec);

    if (auto_compute_normal == true)
        this->autoComputeNormal(glm_pos_vec);
}

template <typename Scalar>
void TriangleRenderUtil::setNormals(const std::vector<Vector<Scalar, 3>> & normals)
{
    if (normals.size() != triangle_num_ * 3)
        throw PhysikaException("error: normal size not match triangle size!");

    std::vector<glm::vec3> glm_normal_vec(normals.size());
    for (unsigned int i = 0; i < normals.size(); ++i)
        glm_normal_vec[i] = { normals[i][0], normals[i][1], normals[i][2] };

    this->updateNormalVBOData(glm_normal_vec);
}

template <typename Scalar>
void TriangleRenderUtil::setTexCoords(const std::vector<Vector<Scalar, 2>> & tex_coords)
{
    if (tex_coords.size() != triangle_num_ * 3)
        throw PhysikaException("error: tex coord size not match triangle size!");

    std::vector<glm::vec2> glm_normal_vec(tex_coords.size());
    for (unsigned int i = 0; i < tex_coords.size(); ++i)
        glm_normal_vec[i] = { tex_coords[i][0], tex_coords[i][1]};

    this->updateTexCoordVBOData(glm_normal_vec);
}

unsigned int TriangleRenderUtil::triangleNum() const
{
    return triangle_num_;
}

void TriangleRenderUtil::draw()
{
    glVerify(glBindVertexArray(triangle_VAO_));
    glVerify(glDrawArrays(GL_TRIANGLES, 0, triangle_num_ * 3));
    glVerify(glBindVertexArray(0));
}

void TriangleRenderUtil::autoComputeNormal(const std::vector<glm::vec3> & glm_pos_vec)
{
    std::vector<glm::vec3> glm_normal_vec;
    for (unsigned int i = 0; i < glm_pos_vec.size(); i += 3)
    {
        const glm::vec3 & v0 = glm_pos_vec[i];
        const glm::vec3 & v1 = glm_pos_vec[i + 1];
        const glm::vec3 & v2 = glm_pos_vec[i + 2];

        glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

        glm_normal_vec.insert(glm_normal_vec.end(), 3, normal);
    }

    this->updateNormalVBOData(glm_normal_vec);
}

void TriangleRenderUtil::bindTriangleVAO()
{
    glVerify(glBindVertexArray(triangle_VAO_));
}

void TriangleRenderUtil::unbindTriangleVAO()
{
    glVerify(glBindVertexArray(0));
}
 
void TriangleRenderUtil::initVAOAndVBOs()
{
    glVerify(glGenVertexArrays(1, &triangle_VAO_));
    glVerify(glBindVertexArray(triangle_VAO_));

    glGenBuffers(1, &triangle_VBO_);
    glBindBuffer(GL_ARRAY_BUFFER, triangle_VBO_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glVerify(glBindVertexArray(0));

    //Note: It's important to not pre config normal&tex_coord VBOs to VAO
    glGenBuffers(1, &normal_VBO_);
    glGenBuffers(1, &tex_coord_VBO_);
}

void TriangleRenderUtil::destoryVAOAndVBOs()
{
    glDeleteVertexArrays(1, &triangle_VAO_);
    glDeleteBuffers(1, &triangle_VBO_);
    glDeleteBuffers(1, &normal_VBO_);
    glDeleteBuffers(1, &tex_coord_VBO_);
}

void TriangleRenderUtil::updateTriangleVBOData(const std::vector<glm::vec3> & glm_pos_vec)
{
    glBindBuffer(GL_ARRAY_BUFFER, triangle_VBO_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * glm_pos_vec.size(), glm_pos_vec.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
    
void TriangleRenderUtil::updateNormalVBOData(const std::vector<glm::vec3> & glm_normal_vec)
{
    glVerify(glBindVertexArray(triangle_VAO_));

    glBindBuffer(GL_ARRAY_BUFFER, normal_VBO_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * glm_normal_vec.size(), glm_normal_vec.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glVerify(glBindVertexArray(0));
}

void TriangleRenderUtil::updateTexCoordVBOData(const std::vector<glm::vec2> & glm_tex_coord_vec)
{
    glVerify(glBindVertexArray(triangle_VAO_));

    glBindBuffer(GL_ARRAY_BUFFER, tex_coord_VBO_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * glm_tex_coord_vec.size(), glm_tex_coord_vec.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glVerify(glBindVertexArray(0));
}

//explicit instantiation
template void TriangleRenderUtil::setTriangles<float, 2>(const std::vector<Vector<float, 2>> & pos_vec, bool auto_compute_normal);
template void TriangleRenderUtil::setTriangles<float, 3>(const std::vector<Vector<float, 3>> & pos_vec, bool auto_compute_normal);
template void TriangleRenderUtil::setTriangles<double, 2>(const std::vector<Vector<double, 2>> & pos_vec, bool auto_compute_normal);
template void TriangleRenderUtil::setTriangles<double, 3>(const std::vector<Vector<double, 3>> & pos_vec, bool auto_compute_normal);

template void TriangleRenderUtil::setTriangles<float, 2>(const std::vector<Vector<float, 2>> & pos_vec, std::vector<unsigned int> & indices, bool auto_compute_normal);
template void TriangleRenderUtil::setTriangles<float, 3>(const std::vector<Vector<float, 3>> & pos_vec, std::vector<unsigned int> & indices, bool auto_compute_normal);
template void TriangleRenderUtil::setTriangles<double, 2>(const std::vector<Vector<double, 2>> & pos_vec, std::vector<unsigned int> & indices, bool auto_compute_normal);
template void TriangleRenderUtil::setTriangles<double, 3>(const std::vector<Vector<double, 3>> & pos_vec, std::vector<unsigned int> & indices, bool auto_compute_normal);

template void TriangleRenderUtil::setNormals<float>(const std::vector<Vector<float, 3>> & normals);
template void TriangleRenderUtil::setNormals<double>(const std::vector<Vector<double, 3>> & normals);

template void TriangleRenderUtil::setTexCoords<float>(const std::vector<Vector<float, 2>> & tex_coords);
template void TriangleRenderUtil::setTexCoords<double>(const std::vector<Vector<double, 2>> & tex_coords);

}//end of namespace Physika