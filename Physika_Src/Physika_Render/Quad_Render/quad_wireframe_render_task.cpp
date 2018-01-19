/*
 * @file quad_wireframe_render_task.h tetrahedron
 * @Basic wireframe render task of quad
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

#include "quad_render_util.h"
#include "quad_wireframe_render_task.h"

namespace Physika{

QuadWireframeRenderTask::QuadWireframeRenderTask(std::shared_ptr<QuadRenderUtil> render_util)
    :LineRenderTask(render_util->getInnerLineRenderUtil()),
    render_util_(std::move(render_util))
{

}    


void QuadWireframeRenderTask::setQuadColors(const std::vector<Color4f>& colors)
{
    if (colors.size() != render_util_->quadNum())
        throw PhysikaException("error: color size not match quad num!");

    std::vector<Color4f> line_colors;
    for (unsigned int i = 0; i < colors.size(); ++i)
        line_colors.insert(line_colors.end(), 4, colors[i]);

    this->setLineColors(line_colors);
}

void QuadWireframeRenderTask::setElementColors(const std::vector<Color4f>& colors)
{
    this->setQuadColors(colors);
}

}//end of namespace Physika