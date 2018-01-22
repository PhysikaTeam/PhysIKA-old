/*
 * @file fluid_point_render_task.cpp
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

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"
#include "Physika_Render/Render_Scene_Config/render_scene_config.h"

#include "fluid_point_render_task.h"
#include "fluid_render_util.h"
#include "fluid_point_render_shader_srcs.h"

namespace Physika{

FluidPointRenderTask::FluidPointRenderTask(std::shared_ptr<FluidRenderUtil> render_util)
    :render_util_(std::move(render_util))
{
    shader_.createFromCStyleString(fluid_point_vert_shader, fluid_point_frag_shader);
}

float FluidPointRenderTask::radius() const
{
    return this->radius_;
}

void FluidPointRenderTask::setRadius(float radius)
{
    this->radius_ = radius;
}

void FluidPointRenderTask::renderTaskImpl()
{
    glVerify(glPushAttrib(GL_ALL_ATTRIB_BITS));

    //enable point sprite
    glEnable(GL_POINT_SPRITE);
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    glEnable(GL_DEPTH_TEST);

    this->configCustomUniforms();

    //real draw
    render_util_->drawByPoint();

    glDisable(GL_POINT_SPRITE);
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);

    glVerify(glPopAttrib());
}

void FluidPointRenderTask::configCustomUniforms()
{
    bool show_density = false;
    bool use_shadow_map = true;

    int mode = 0;
    if (show_density == true)
        mode = 1;
    if (use_shadow_map == false)
        mode = 2;
    openGLSetCurBindShaderInt("mode", mode);

    openGLSetCurBindShaderFloat("pointRadius", this->radius_);

    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
    unsigned int screen_width = render_scene_config.screenWidth();
    unsigned int screen_height = render_scene_config.screenHeight();
    float        screen_aspect = static_cast<float>(screen_width) / screen_height;


    float point_scale = screen_width / screen_aspect * (1.0f / tanf(this->fov_*0.5f));
    openGLSetCurBindShaderFloat("pointScale", point_scale);
    
    float colors[32] = {
                          0.0f,   0.5f,   1.0f,   1.0f,
                          0.797f, 0.354f, 0.000f, 1.0f,
                          0.092f, 0.465f, 0.820f, 1.0f,
                          0.000f, 0.349f, 0.173f, 1.0f,
                          0.875f, 0.782f, 0.051f, 1.0f,
                          0.000f, 0.170f, 0.453f, 1.0f,
                          0.673f, 0.111f, 0.000f, 1.0f,
                          0.612f, 0.194f, 0.394f, 1.0f,
                        };

    openGLSetCurBindShaderFloat4V("colors", 8, colors);

    //to delete
    this->configFakeLightUniforms();
}

void FluidPointRenderTask::configFakeLightUniforms(bool reverse_light_dir)
{
    glm::vec3 light_pos = { 100, 150, 0 };
    glm::vec3 light_target = { 0, 0, 0 };
    glm::vec3 light_dir = glm::normalize(light_target - light_pos);
    glm::vec3 light_up = { 0.0f, 1.0f, 0.0f };

    glm::mat4 light_proj_mat = glm::perspective(glm::radians(45.0f), 1.0f, 1.0f, 1000.0f);
    glm::mat4 light_model_view_mat = glm::lookAt(light_pos, light_target, light_up);
    glm::mat4 light_transform_mat = light_proj_mat*light_model_view_mat;


    openGLSetCurBindShaderMat4("lightTransform", light_transform_mat);
    openGLSetCurBindShaderVec3("lightPos", light_pos);
    openGLSetCurBindShaderVec3("lightDir", reverse_light_dir ? -light_dir : light_dir);
    openGLSetCurBindShaderFloat("spotMin", 0.0f);
    openGLSetCurBindShaderFloat("spotMax", 0.5f);
}
    
}//end of namespace physika