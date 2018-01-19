/*
 * @file tetrahedron_wireframe_render_task.h 
 * @Basic wireframe render task of tetrahedron
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

#include "tetrahedron_render_util.h"
#include "tetrahedron_wireframe_render_task.h"

namespace Physika{

TetrahedronWireframeRenderTask::TetrahedronWireframeRenderTask(std::shared_ptr<TetrahedronRenderUtil> render_util)
    :TriangleWireframeRenderTask(render_util->getInnerTriangleRenderUtil()),
    render_util_(std::move(render_util))
{

}    

TetrahedronWireframeRenderTask::~TetrahedronWireframeRenderTask()
{
    //do nothing
}

void TetrahedronWireframeRenderTask::setTetColors(const std::vector<Color4f>& colors)
{
    if (colors.size() != render_util_->tetrahedronNum())
        throw PhysikaException("error: color size not match tet num!");

    std::vector<Color4f> triangle_colors;
    for (unsigned int i = 0; i < colors.size(); ++i)
        triangle_colors.insert(triangle_colors.end(), 4, colors[i]);

    this->setTriangleColors(triangle_colors);
}

void TetrahedronWireframeRenderTask::setElementColors(const std::vector<Color4f>& colors)
{
    this->setElementColors(colors);
}

}//end of namespace Physika