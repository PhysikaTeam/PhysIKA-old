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

#pragma once

#include <vector>
#include <glm/fwd.hpp>

#include "Physika_Render/Color/color.h"

namespace Physika{

class VertexColorAttribConfig
{
public:
    VertexColorAttribConfig();
    ~VertexColorAttribConfig();

    //disable copy
    VertexColorAttribConfig(const VertexColorAttribConfig &) = delete;
    VertexColorAttribConfig & operator = (const VertexColorAttribConfig &) = delete;

    //enable move
    VertexColorAttribConfig(VertexColorAttribConfig && rhs) noexcept;
    VertexColorAttribConfig & operator = (VertexColorAttribConfig && rhs) noexcept;

    void setUniformColor(const Color4f & color);
    void setColors(const std::vector<Color4f> & colors);
    void setColors(const std::vector<glm::vec3> & colors);

    //Note: throw exception if color size not match valid_num and not use uniform color
    void config(unsigned int valid_num) const;
    unsigned int colorNum() const;

private:
    void initColorVBO();
    void destroyColorVBO();

    void updateColorVBOData(const std::vector<glm::vec3> & glm_col_vec);

private:
    unsigned int color_num_ = 1;
    bool use_uniform_color_ = true;
    Color4f uniform_color_ = Color4f::Red();
    unsigned int color_VBO_;
};
    
}//end of namespace Physika