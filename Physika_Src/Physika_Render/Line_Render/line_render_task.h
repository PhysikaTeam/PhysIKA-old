/*
 * @file line_render_task.h 
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

#pragma once

#include <memory>
#include <vector>

#include "Physika_Render/Color/color.h"
#include "Physika_Render/Render_Task_Base/render_task_base.h"
#include "Physika_Render/Utilities/vertex_color_attrib_config.h"

namespace Physika{

class LineRenderUtil;

class LineRenderTask: public RenderTaskBase
{
public:
    explicit LineRenderTask(std::shared_ptr<LineRenderUtil> render_util);
    ~LineRenderTask();

    //disable copy
    LineRenderTask(const LineRenderTask &) = delete;
    LineRenderTask & operator = (const LineRenderTask &) = delete;

    void setUniformColor(const Color4f & color);
    void setVertexColors(const std::vector<Color4f> & colors);          //set two point colors for each line
    void setLineColors(const std::vector<Color4f> & colors);
    virtual void setElementColors(const std::vector<Color4f> & colors); //designed for derived classes

    void setLineWidth(float line_width);
    float lineWidth() const;

private:
    void renderTaskImpl() override;
    void configColorToLineVAO();

private:
    std::shared_ptr<LineRenderUtil> render_util_;

    VertexColorAttribConfig color_config_;

    float line_widht_ = 2.0;
};
    
}