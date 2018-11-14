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

#pragma once

#include <memory>

#include "Physika_Render/Quad_Render/quad_wireframe_render_task.h"


namespace Physika{

class CubeRenderUtil;

class CubeWireframeRenderTask: public QuadWireframeRenderTask
{
public:
    explicit CubeWireframeRenderTask(std::shared_ptr<CubeRenderUtil> render_util);
    ~CubeWireframeRenderTask() = default;

    //disable copy
    CubeWireframeRenderTask(const CubeWireframeRenderTask &) = delete;
    CubeWireframeRenderTask & operator = (const CubeWireframeRenderTask &) = delete;

    void setCubeColors(const std::vector<Color4f> & colors);
    void setElementColors(const std::vector<Color4f> & colors) override;

private:
    std::shared_ptr<CubeRenderUtil> render_util_;
};

}//end of namespace Physika    