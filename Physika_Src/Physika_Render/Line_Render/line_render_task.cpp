/*
 * @file line_render_task.cpp
 * @Basic render task of line
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

#include "line_render_util.h"
#include "line_render_shader_srcs.h"
#include "line_render_task.h"


namespace Physika{

LineRenderTask::LineRenderTask(std::shared_ptr<LineRenderUtil> render_util)
    :render_util_(std::move(render_util))
{
    shader_.createFromCStyleString(line_render_vertex_shader, line_render_frag_shader);
}    

LineRenderTask::~LineRenderTask()
{

}

void LineRenderTask::setUniformColor(const Color4f & color)
{
    color_config_.setUniformColor(color);
}

void LineRenderTask::setVertexColors(const std::vector<Color4f> & colors)
{
    if (colors.size() != render_util_->lineNum() * 2)
        throw PhysikaException("error: color size not match!");

    color_config_.setColors(colors);
}

void LineRenderTask::setLineColors(const std::vector<Color4f> & colors)
{
    if (colors.size() != render_util_->lineNum())
        throw PhysikaException("error: color size not match!");

    std::vector<glm::vec3> glm_col_vec;
    for (const Color4f & col : colors)
        glm_col_vec.insert(glm_col_vec.end(), 2, convertCol3(col));

    color_config_.setColors(glm_col_vec);
}

void LineRenderTask::setElementColors(const std::vector<Color4f> & colors)
{
    this->setLineColors(colors);
}

void LineRenderTask::setLineWidth(float line_width)
{
    line_widht_ = line_width;
}

float LineRenderTask::lineWidth() const
{
    return line_widht_;
}

void LineRenderTask::renderTaskImpl()
{
    this->configColorToLineVAO();

    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glLineWidth(line_widht_);

    render_util_->draw();

    glPopAttrib();
}

void LineRenderTask::configColorToLineVAO()
{
    //bind line VAO
    render_util_->bindLineVAO();

    //bind color to line VAO
    color_config_.config(render_util_->lineNum() * 2);

    //unbind line VAO
    render_util_->unbindLineVAO();
}

}//end of namespace Physika