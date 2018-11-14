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
#include "triangle_custom_color_render_task_base.h"

namespace Physika{

class TriangleRenderUtil;

class TriangleSolidRenderTask: public TriangleCustomColorRenderTaskBase
{
public:
    explicit TriangleSolidRenderTask(std::shared_ptr<TriangleRenderUtil> render_util);
    ~TriangleSolidRenderTask() = default;

    //disable copy
    TriangleSolidRenderTask(const TriangleSolidRenderTask &) = delete;
    TriangleSolidRenderTask & operator = (const TriangleSolidRenderTask &) = delete;

    void enableUseLight();
    void disableUseLight();
    bool isUseLight() const;

private:
    void customConfigs() override;

private:
    bool use_light_ = true;
};
    
}