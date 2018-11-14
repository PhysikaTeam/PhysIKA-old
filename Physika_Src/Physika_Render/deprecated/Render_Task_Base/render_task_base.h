/*
 * @file render_task_base.h 
 * @Basic render task, all other render task class inherit from this class.
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

#pragma  once

#include "Physika_Render/OpenGL_Shaders/shader.h"
#include "Physika_Core/Transform/transform_3d.h"

namespace Physika{


enum class RenderTaskType
{
    NORMAL_RENDER_TASK,
    SCREEN_BASED_RENDER_TASK
};


class RenderTaskBase
{
public:
    virtual ~RenderTaskBase() = default;

    void renderTask();

    virtual RenderTaskType type() const;  //default to RenderTaskType::NORMAL_RENDER_TASK

    void enableBindShader();
    void disableBindShader();
    bool isEnableBindShader() const;

    const Transform3f & transform()const;
    void setTransform(const Transform3f & transform);

private:
    virtual void renderTaskImpl() = 0;

protected:
    Shader shader_;  //enable creation in derived class

private:
    bool enable_bind_shader_ = true; //hook to enable/disable bind shader
    Transform3f transform_;
};
    
}//end of namespace Physika