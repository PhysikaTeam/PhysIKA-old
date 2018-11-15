/*
* @file point_vector_render_shader_srcs.h
* @Brief GLSL shaders for point vector render
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

static const char * point_vector_vertex_shader = "#version 330 core\n" STRINGIFY(

layout(location = 0) in vec3 vert_pos;
layout(location = 4) in vec3 vert_vector;

out vec3 geo_vert_vector;

uniform mat4 proj_trans;
uniform mat4 view_trans;
uniform mat4 model_trans;

void main()
{
    gl_Position = proj_trans * view_trans * model_trans * vec4(vert_pos, 1.0);

    mat3 normal_mat = mat3(transpose(inverse(view_trans * model_trans)));
    geo_vert_vector = vec3(proj_trans * vec4(normal_mat * vert_vector, 1.0));
}

);

static const char * point_vector_geo_shader = "#version 330 core\n" STRINGIFY(

layout (points) in;
layout (line_strip, max_vertices = 2) out;

in vec3 geo_vert_vector[];
out vec3 frag_vert_col;

uniform float scale_factor;

void main()
{
    frag_vert_col = geo_vert_vector[0];

    gl_Position = gl_in[0].gl_Position;
    EmitVertex();

    gl_Position = gl_in[0].gl_Position + vec4(geo_vert_vector[0], 0.0) * scale_factor;
    EmitVertex();

    EndPrimitive();
}

);

static const char * point_vector_frag_shader = "#version 330 core\n" STRINGIFY(

in vec3 frag_vert_col;
out vec4 frag_color;

uniform bool use_point_vector_col;
uniform vec3 col;

void main()
{
    if (use_point_vector_col)
        frag_color = vec4(frag_vert_col, 1.0);
    else
        frag_color = vec4(col, 1.0);
}

);

}//end of namespace Physika