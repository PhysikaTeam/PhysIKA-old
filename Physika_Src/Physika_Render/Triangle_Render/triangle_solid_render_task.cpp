/*
 * @file triangle_solid_render_task.h 
 * @Basic solid render task of triangle
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

#include "Physika_Core/Utilities/glm_utilities.h"
#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"

#include "triangle_render_util.h"
#include "triangle_solid_render_shader_srcs.h"
#include "triangle_solid_render_task.h"


namespace Physika{

TriangleSolidRenderTask::TriangleSolidRenderTask(std::shared_ptr<TriangleRenderUtil> render_util)
    :TriangleCustomColorRenderTaskBase(std::move(render_util))
{
    shader_.createFromCStyleString(triangle_solid_render_vertex_shader, triangle_solid_render_frag_shader);
}    



void TriangleSolidRenderTask::enableUseLight()
{
    use_light_ = true;
}

void TriangleSolidRenderTask::disableUseLight()
{
    use_light_ = false;
}

bool TriangleSolidRenderTask::isUseLight() const
{
    return use_light_;
}

void TriangleSolidRenderTask::customConfigs()
{
    openGLSetCurBindShaderBool("use_light", use_light_);
}

}//end of namespace Physika