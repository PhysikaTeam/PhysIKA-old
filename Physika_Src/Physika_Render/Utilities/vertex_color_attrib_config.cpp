/*
 * @file vertex_color_attrib_config.h 
 * @Basic class VertexColorAttribConfig
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

#include "vertex_color_attrib_config.h"

namespace Physika{

VertexColorAttribConfig::VertexColorAttribConfig()
{
    this->initColorVBO();
}

VertexColorAttribConfig::~VertexColorAttribConfig()
{
    this->destroyColorVBO();
}

VertexColorAttribConfig::VertexColorAttribConfig(VertexColorAttribConfig && rhs) noexcept
{
    this->color_num_ = rhs.color_num_;
    this->use_uniform_color_ = rhs.use_uniform_color_;
    this->uniform_color_ = rhs.uniform_color_;
    this->color_VBO_ = rhs.color_VBO_;

    rhs.color_VBO_ = 0;
}

VertexColorAttribConfig & VertexColorAttribConfig::operator = (VertexColorAttribConfig && rhs) noexcept
{
    this->color_num_ = rhs.color_num_;
    this->use_uniform_color_ = rhs.use_uniform_color_;
    this->uniform_color_ = rhs.uniform_color_;
    this->color_VBO_ = rhs.color_VBO_;

    rhs.color_VBO_ = 0;

    return *this;
}

void VertexColorAttribConfig::setUniformColor(const Color4f & color)
{
    use_uniform_color_ = true;
    color_num_ = 1;
    uniform_color_ = color;
}

void VertexColorAttribConfig::setColors(const std::vector<Color4f> & colors)
{
    use_uniform_color_ = false;
    color_num_ = colors.size();

    std::vector<glm::vec3> glm_col_vec(colors.size());
    for (unsigned int i = 0; i < colors.size(); ++i)
        glm_col_vec[i] = convertCol3(colors[i]);

    this->updateColorVBOData(glm_col_vec);
}

void VertexColorAttribConfig::setColors(const std::vector<glm::vec3> & colors)
{
    use_uniform_color_ = false;
    color_num_ = colors.size();

    this->updateColorVBOData(colors);
}

void VertexColorAttribConfig::config(unsigned int valid_num) const
{
    if(use_uniform_color_ == true)
    {
        glVerify(glDisableVertexAttribArray(3));

        glm::vec3 glm_col = convertCol3(uniform_color_);
        glVerify(glVertexAttrib3fv(3, glm::value_ptr(glm_col)));
    }
    else
    {
        if (valid_num != color_num_)
            throw PhysikaException("error: color size not match!");

        glBindBuffer(GL_ARRAY_BUFFER, color_VBO_);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(3);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

unsigned int VertexColorAttribConfig::colorNum() const
{
    return color_num_;
}

void VertexColorAttribConfig::initColorVBO()
{
    glGenBuffers(1, &color_VBO_);
    glBindBuffer(GL_ARRAY_BUFFER, color_VBO_);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void VertexColorAttribConfig::destroyColorVBO()
{
    glDeleteBuffers(1, &color_VBO_);
    color_VBO_ = 0;
}

void VertexColorAttribConfig::updateColorVBOData(const std::vector<glm::vec3> & glm_col_vec)
{
    glBindBuffer(GL_ARRAY_BUFFER, color_VBO_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * glm_col_vec.size(), glm_col_vec.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
    
}//end of namespace Physika