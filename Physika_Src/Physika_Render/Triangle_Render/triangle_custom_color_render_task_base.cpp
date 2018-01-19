/*
 * @file triangle_solid_render_task.h 
 * @Basic solid render task of triangle
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

#include "triangle_render_util.h"
#include "triangle_custom_color_render_task_base.h"


namespace Physika{

TriangleCustomColorRenderTaskBase::TriangleCustomColorRenderTaskBase(std::shared_ptr<TriangleRenderUtil> render_util)
    :render_util_(std::move(render_util))
{
    
}    

void TriangleCustomColorRenderTaskBase::enableUseCustomColor()
{
    use_custom_color_ = true;
}

void TriangleCustomColorRenderTaskBase::disableUseCustomColor()
{
    use_custom_color_ = false;
}

bool TriangleCustomColorRenderTaskBase::isUseCustomColor() const
{
    return use_custom_color_;
}

void TriangleCustomColorRenderTaskBase::setUniformColor(const Color4f & color)
{
    color_config_.setUniformColor(color);
}

void TriangleCustomColorRenderTaskBase::setVertexColors(const std::vector<Color4f> & colors)
{
    if(colors.size() != render_util_->triangleNum() * 3)
        throw PhysikaException("error: color size not match!");

    color_config_.setColors(colors);
}

void TriangleCustomColorRenderTaskBase::setTriangleColors(const std::vector<Color4f> & colors)
{
    if (colors.size() != render_util_->triangleNum())
        throw PhysikaException("error: color size not match!");

    std::vector<glm::vec3> glm_col_vec;
    for (const Color4f & col : colors)
        glm_col_vec.insert(glm_col_vec.end(), 3, convertCol3(col));

    color_config_.setColors(glm_col_vec);
}

void TriangleCustomColorRenderTaskBase::setElementColors(const std::vector<Color4f> & colors)
{
    this->setTriangleColors(colors);
}


void TriangleCustomColorRenderTaskBase::renderTaskImpl()
{
    this->configColorToTriangleVAO();

    openGLSetCurBindShaderBool("use_custom_color", use_custom_color_);

    glPushAttrib(GL_ALL_ATTRIB_BITS);

    this->customConfigs();
    render_util_->draw();

    glPopAttrib();
}

void TriangleCustomColorRenderTaskBase::configColorToTriangleVAO()
{
    //bind triangle VAO
    render_util_->bindTriangleVAO();

    //bind color VBO to point VAO
    color_config_.config(render_util_->triangleNum() * 3);

    //unbind triangle VAO
    render_util_->unbindTriangleVAO();
}

}//end of namespace Physika