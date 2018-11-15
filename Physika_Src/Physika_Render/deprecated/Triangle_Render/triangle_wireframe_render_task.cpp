/*
 * @file triangle_wireframe_render_task.h 
 * @Basic wireframe render task of triangle
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
#include "triangle_wireframe_render_shader_srcs.h"
#include "triangle_wireframe_render_task.h"


namespace Physika{

TriangleWireframeRenderTask::TriangleWireframeRenderTask(std::shared_ptr<TriangleRenderUtil> render_util)
    :TriangleCustomColorRenderTaskBase(std::move(render_util))
{
    shader_.createFromCStyleString(triangle_wireframe_render_vertex_shader, triangle_wireframe_render_frag_shader);
}    


void TriangleWireframeRenderTask::setLineWidth(float line_width)
{
    line_widht_ = line_width;
}

float TriangleWireframeRenderTask::lineWidth() const
{
    return line_widht_;
}

void TriangleWireframeRenderTask::customConfigs()
{
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glLineWidth(line_widht_);
}

}//end of namespace Physika