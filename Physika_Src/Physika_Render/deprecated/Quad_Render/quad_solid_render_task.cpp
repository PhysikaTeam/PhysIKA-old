/*
 * @file quad_solid_render_task.h 
 * @Basic solid render task of quad
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

#include "quad_render_util.h"
#include "quad_solid_render_task.h"

namespace Physika{

QuadSolidRenderTask::QuadSolidRenderTask(std::shared_ptr<QuadRenderUtil> render_util)
    :TriangleSolidRenderTask(render_util->getInnerTriangleRenderUtil()),
    render_util_(std::move(render_util))
{

}    

void QuadSolidRenderTask::setQuadColors(const std::vector<Color4f>& colors)
{
    if (colors.size() != render_util_->quadNum())
        throw PhysikaException("error: color size not match quad num!");

    std::vector<Color4f> triangle_colors;
    for (unsigned int i = 0; i < colors.size(); ++i)
        triangle_colors.insert(triangle_colors.end(), 2, colors[i]);

    this->setTriangleColors(triangle_colors);
}

void QuadSolidRenderTask::setElementColors(const std::vector<Color4f>& colors)
{
    this->setQuadColors(colors);
}

}//end of namespace Physika