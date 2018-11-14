/*
 * @file point_vector_attrib_config.h 
 * @Basic class PointVectorAttribConfig
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

#include "Physika_Core/Utilities/glm_utilities.h"
#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"

#include "point_vector_attrib_config.h"

namespace Physika{

PointVectorAttribConfig::PointVectorAttribConfig()
{
    this->initPointVectorVBO();
}

PointVectorAttribConfig::~PointVectorAttribConfig()
{
    this->destroyPointVectorVBO();
}

PointVectorAttribConfig::PointVectorAttribConfig(PointVectorAttribConfig && rhs) noexcept
{
    this->point_vector_num_ = rhs.point_vector_num_;
    this->use_uniform_point_vector_ = rhs.use_uniform_point_vector_;
    this->uniform_point_vector_ = rhs.uniform_point_vector_;
    this->point_vector_VBO_ = rhs.point_vector_VBO_;

    rhs.point_vector_VBO_ = 0;
}

PointVectorAttribConfig & PointVectorAttribConfig::operator = (PointVectorAttribConfig && rhs) noexcept
{
    this->point_vector_num_ = rhs.point_vector_num_;
    this->use_uniform_point_vector_ = rhs.use_uniform_point_vector_;
    this->uniform_point_vector_ = rhs.uniform_point_vector_;
    this->point_vector_VBO_ = rhs.point_vector_VBO_;

    rhs.point_vector_VBO_ = 0;

    return *this;
}

template <typename Scalar>
void PointVectorAttribConfig::setUniformPointVector(const Vector<Scalar, 3> & point_vector)
{
    use_uniform_point_vector_ = true;
    point_vector_num_ = 1;
    uniform_point_vector_ = Vector3f(point_vector[0], point_vector[1], point_vector[2]);
}

template <typename Scalar>
void PointVectorAttribConfig::setPointVectors(const std::vector<Vector<Scalar, 3>> & point_vectors)
{
    use_uniform_point_vector_ = false;
    point_vector_num_ = point_vectors.size();

    std::vector<glm::vec3> glm_col_vec(point_vectors.size());
    for (unsigned int i = 0; i < point_vectors.size(); ++i)
        glm_col_vec[i] = convert(point_vectors[i]);

    this->updatePointVectorVBOData(glm_col_vec);
}

void PointVectorAttribConfig::setPointVectors(const std::vector<glm::vec3> & point_vectors)
{
    use_uniform_point_vector_ = false;
    point_vector_num_ = point_vectors.size();

    this->updatePointVectorVBOData(point_vectors);
}

void PointVectorAttribConfig::config(unsigned int valid_num) const
{
    if(use_uniform_point_vector_ == true)
    {
        glVerify(glDisableVertexAttribArray(4));

        glm::vec3 glm_vector = convert(uniform_point_vector_);
        glVerify(glVertexAttrib3fv(4, glm::value_ptr(glm_vector)));
    }
    else
    {
        if (valid_num != point_vector_num_)
            throw PhysikaException("error: point_vector size not match!");

        glBindBuffer(GL_ARRAY_BUFFER, point_vector_VBO_);
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(4);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

unsigned int PointVectorAttribConfig::point_vectorNum() const
{
    return point_vector_num_;
}

void PointVectorAttribConfig::initPointVectorVBO()
{
    glGenBuffers(1, &point_vector_VBO_);
    glBindBuffer(GL_ARRAY_BUFFER, point_vector_VBO_);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void PointVectorAttribConfig::destroyPointVectorVBO()
{
    glDeleteBuffers(1, &point_vector_VBO_);
    point_vector_VBO_ = 0;
}

void PointVectorAttribConfig::updatePointVectorVBOData(const std::vector<glm::vec3> & glm_col_vec)
{
    glBindBuffer(GL_ARRAY_BUFFER, point_vector_VBO_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * glm_col_vec.size(), glm_col_vec.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

template void PointVectorAttribConfig::setUniformPointVector(const Vector<float, 3> & point_vector);
template void PointVectorAttribConfig::setUniformPointVector(const Vector<double, 3> & point_vector);
template void PointVectorAttribConfig::setPointVectors(const std::vector<Vector<float, 3>> & point_vector);
template void PointVectorAttribConfig::setPointVectors(const std::vector<Vector<double, 3>> & point_vector);
    
}//end of namespace Physika