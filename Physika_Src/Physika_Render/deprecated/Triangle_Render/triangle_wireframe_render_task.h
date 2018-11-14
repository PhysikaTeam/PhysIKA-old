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

#pragma once

#include <memory>
#include <vector>

#include "Physika_Render/Color/color.h"
#include "Physika_Render/Render_Task_Base/render_task_base.h"
#include "Physika_Render/Utilities/vertex_color_attrib_config.h"

#include "triangle_custom_color_render_task_base.h"

namespace Physika{

class TriangleRenderUtil;

class TriangleWireframeRenderTask: public TriangleCustomColorRenderTaskBase
{
public:
    explicit TriangleWireframeRenderTask(std::shared_ptr<TriangleRenderUtil> render_util);
    ~TriangleWireframeRenderTask() = default;

    //disable copy
    TriangleWireframeRenderTask(const TriangleWireframeRenderTask &) = delete;
    TriangleWireframeRenderTask & operator = (const TriangleWireframeRenderTask &) = delete;

    void setLineWidth(float triangle_width);
    float lineWidth() const;

private:
    void customConfigs() override;

private:
    float line_widht_ = 2.0;
};
    
}