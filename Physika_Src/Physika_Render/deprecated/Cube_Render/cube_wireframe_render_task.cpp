/*
 * @file cube_wireframe_render_task.h 
 * @Basic wireframe render task of cube
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

#include "cube_render_util.h"
#include "cube_wireframe_render_task.h"

namespace Physika{

CubeWireframeRenderTask::CubeWireframeRenderTask(std::shared_ptr<CubeRenderUtil> render_util)
    :QuadWireframeRenderTask(render_util->getInnerQuadRenderUtil()),
    render_util_(std::move(render_util))
{

}    

void CubeWireframeRenderTask::setCubeColors(const std::vector<Color4f>& colors)
{
    if (colors.size() != render_util_->cubeNum())
        throw PhysikaException("error: color size not match cube num!");

    std::vector<Color4f> quad_colors;
    for (unsigned int i = 0; i < colors.size(); ++i)
        quad_colors.insert(quad_colors.end(), 6, colors[i]);

    this->setQuadColors(quad_colors);
}

void CubeWireframeRenderTask::setElementColors(const std::vector<Color4f>& colors)
{
    this->setCubeColors(colors);
}

}//end of namespace Physika