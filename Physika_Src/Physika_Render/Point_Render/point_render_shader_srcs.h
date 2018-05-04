/*
* @file point_render_shader_srcs.h
* @Brief GLSL shaders for point render
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

namespace Physika {

#define STRINGIFY(A) #A

static const char * point_render_vertex_shader = "#version 330 core\n" STRINGIFY(

layout(location = 0) in vec3 vert_pos;
layout(location = 3) in vec3 vert_col;

out vec3 frag_vert_col;

uniform mat4 proj_trans;
uniform mat4 view_trans;
uniform mat4 model_trans;

uniform bool use_point_sprite;
uniform float point_size;
uniform float point_scale;

void main()
{
    frag_vert_col = vert_col;

    gl_Position = proj_trans * view_trans * model_trans * vec4(vert_pos, 1.0);
    if(use_point_sprite)
    {
        gl_PointSize = point_scale * point_size / gl_Position.w;
    }
}

);

static const char * point_render_frag_shader = "#version 330 core\n" STRINGIFY(

in vec3 frag_vert_col;
out vec4 frag_color;

uniform bool use_point_sprite;

void main()
{
    if(use_point_sprite)
    {
        vec3 normal;
        //normal.xy = gl_PointCoord.xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
        normal.xy = gl_PointCoord.xy * 2.0 -1.0;
        float mag = dot(normal.xy, normal.xy);
        if (mag > 1.0) discard;   
        normal.z = sqrt(1.0 - mag);

        vec3 light_dir = vec3(0, 1, 0);
        float diffuse_factor = max(dot(light_dir, normal), 0);
        vec3 diffuse = diffuse_factor * frag_vert_col;

        diffuse = vec3(normal.z) * frag_vert_col;
        frag_color = vec4(diffuse, 1.0);
    }
    else
    {
        frag_color = vec4(frag_vert_col, 1.0);
    }
}

);

}//end of namespace Physika