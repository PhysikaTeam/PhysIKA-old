/*
 * @file render_task_base.cpp
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



#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"
#include "Physika_Core/Transform/transform_3d.h"
#include "Physika_Render/OpenGL_Shaders/shader.h"
#include "Physika_Render/Global_Unifrom_Config/global_uniform_config.h"

#include "render_task_base.h"

namespace Physika{

void RenderTaskBase::renderTask()
{
    if (enable_bind_shader_ == true)
        shader_.bindAndConfigBasicUniforms();

    //config global uniforms
    GlobalUniformConfig & global_uniform_config = GlobalUniformConfig::getSingleton();
    global_uniform_config.configGlobalUniforms();

    //set model transform
    const Matrix4f & model_trans = transform_.transformMatrix();
    openGLSetCurBindShaderMat4("model_trans", model_trans);

    glVerify(glPushAttrib(GL_ALL_ATTRIB_BITS));

    renderTaskImpl();

    glVerify(glPopAttrib());

    if (enable_bind_shader_ == true)
        shader_.unBind();
}

RenderTaskType RenderTaskBase::type() const
{
    return RenderTaskType::NORMAL_RENDER_TASK;
}

void RenderTaskBase::enableBindShader()
{
    enable_bind_shader_ = true;
}

void RenderTaskBase::disableBindShader()
{
    enable_bind_shader_ = false;
}

bool RenderTaskBase::isEnableBindShader() const
{
    return enable_bind_shader_;
}

const Transform3f & RenderTaskBase::transform()const
{
    return this->transform_;
}

void RenderTaskBase::setTransform(const Transform3f & transform)
{
    this->transform_ = transform;
}

}//end of namespace Physika