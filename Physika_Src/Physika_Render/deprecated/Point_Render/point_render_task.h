/*
 * @file point_render_task.h 
 * @Basic render task of point
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

class PointRenderUtil;

class PointRenderTask: public RenderTaskBase
{
public:
    explicit PointRenderTask(std::shared_ptr<PointRenderUtil> render_util);
    ~PointRenderTask();

    //disable copy
    PointRenderTask(const PointRenderTask &) = delete;
    PointRenderTask & operator = (const PointRenderTask &) = delete;

    void setUniformColor(const Color4f & color);
    void setPointColors(const std::vector<Color4f> & colors);

    void setPointSize(float point_size);
    float pointSize() const;

    void setPointScaleForPointSprite(float point_scale);
    float pointScaleForPointSprite() const;

    void enableUsePointSprite();
    void disableUsePointSprite();
    bool isUsePointSprite() const;

private:
    void renderTaskImpl() override;
    void configColorToPointVAO();

private:
    std::shared_ptr<PointRenderUtil> render_util_;

    VertexColorAttribConfig color_config_;

    bool use_point_sprite_ = true;

    float point_size_ = 2.0f;
    float point_scale_ = 10.0f; //for point sprite
};
    
}
