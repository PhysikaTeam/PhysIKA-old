/*
 * @file fluid_point_render_task.h 
 * @Basic point render task of fluid
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
#include "Physika_Render/Render_Task_Base/render_task_base.h"

namespace Physika{

class FluidRenderUtil;

class FluidPointRenderTask: public RenderTaskBase
{
public:
    FluidPointRenderTask(std::shared_ptr<FluidRenderUtil> render_util);

    float radius() const;
    void setRadius(float radius);
    
private:
    void renderTaskImpl() override;

    void configCustomUniforms();
    void configFakeLightUniforms(bool reverse_light_dir = false); //to delete
    
    
private:
    std::shared_ptr<FluidRenderUtil> render_util_;

    float radius_ = 0.01f;
    float fov_ = 3.14159f / 4.0f;
};
    
}//end of namespace Physika