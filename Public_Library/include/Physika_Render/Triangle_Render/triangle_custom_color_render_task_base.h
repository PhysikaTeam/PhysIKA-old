/*
 * @file triangle_solid_render_task.h 
 * @Basic render task of triangle
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

class TriangleRenderUtil;

class TriangleCustomColorRenderTaskBase: public RenderTaskBase
{
public:
    explicit TriangleCustomColorRenderTaskBase(std::shared_ptr<TriangleRenderUtil> render_util);
    ~TriangleCustomColorRenderTaskBase() = default;

    //disable copy
    TriangleCustomColorRenderTaskBase(const TriangleCustomColorRenderTaskBase &) = delete;
    TriangleCustomColorRenderTaskBase & operator = (const TriangleCustomColorRenderTaskBase &) = delete;

    void enableUseCustomColor();
    void disableUseCustomColor();
    bool isUseCustomColor() const;

    void setUniformColor(const Color4f & color);
    void setVertexColors(const std::vector<Color4f> & colors);
    void setTriangleColors(const std::vector<Color4f> & colors);
    virtual void setElementColors(const std::vector<Color4f> & colors);

protected:
    void renderTaskImpl() final;  //Note: we intentionlly change the renderTaskImpl access mode from "private" to "protected",
                                  //      which enable us to further change it from "protected" to "public" by "using ***Base::renderTaskImpl" after the "public:" descriptor.
                                  //      We specially do this for volumetricMesh***RenderTask(s).

                                  //Note: we have no necessary do this for msvc(VS) compiler as it's legal to "using ***Base::renderTaskImpl" although it's "private".

private:
    virtual void customConfigs() = 0;
    void configColorToTriangleVAO();

private:
    std::shared_ptr<TriangleRenderUtil> render_util_;

    bool use_custom_color_ = true;
    VertexColorAttribConfig color_config_;
};
    
}