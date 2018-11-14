/*
 * @file surface_mesh_point_render_task.h 
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

#include <glm/glm.hpp>
#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"

#include "point_render_util.h"
#include "point_render_shader_srcs.h"
#include "point_render_task.h"

namespace Physika{

PointRenderTask::PointRenderTask(std::shared_ptr<PointRenderUtil> render_util)
    :render_util_(std::move(render_util))
{
    shader_.createFromCStyleString(point_render_vertex_shader, point_render_frag_shader);
}    

PointRenderTask::~PointRenderTask()
{
}

void PointRenderTask::setUniformColor(const Color4f & color)
{
    color_config_.setUniformColor(color);
}

void PointRenderTask::setPointColors(const std::vector<Color4f> & colors)
{
    if (colors.size() != render_util_->pointNum())
        throw PhysikaException("error: color size not match!");

    color_config_.setColors(colors);
}

void PointRenderTask::setPointSize(float point_size)
{
    point_size_ = point_size;
}

float PointRenderTask::pointSize() const
{
    return point_size_;
}

void PointRenderTask::setPointScaleForPointSprite(float point_scale)
{
    point_scale_ = point_scale;
}

float PointRenderTask::pointScaleForPointSprite() const
{
    return point_scale_;
}

void PointRenderTask::enableUsePointSprite()
{
    use_point_sprite_ = true;
}

void PointRenderTask::disableUsePointSprite()
{
    use_point_sprite_ = false;
}

bool PointRenderTask::isUsePointSprite() const
{
    return use_point_sprite_;
}

void PointRenderTask::renderTaskImpl()
{
    this->configColorToPointVAO();

    glPushAttrib(GL_ALL_ATTRIB_BITS);

    if(use_point_sprite_)
    {
        glEnable(GL_POINT_SPRITE);
        glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
        glEnable(GL_PROGRAM_POINT_SIZE);

        openGLSetCurBindShaderFloat("point_size", point_size_);
        openGLSetCurBindShaderFloat("point_scale", point_scale_);
        openGLSetCurBindShaderBool("use_point_sprite", true);
    }
   else
    {
        glPointSize(point_size_);
    }
    render_util_->draw();

    glPopAttrib();
}

void PointRenderTask::configColorToPointVAO()
{
    //bind point VAO
    render_util_->bindPointVAO();

    //config color to point VAO
    color_config_.config(render_util_->pointNum());
    
    //unbind point VAO
    render_util_->unbindPointVAO();
}

}//end of namespace Physika