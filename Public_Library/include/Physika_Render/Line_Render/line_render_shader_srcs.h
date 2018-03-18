/*
* @file line_render_shader_srcs.h
* @Brief GLSL shaders for line render
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

static const char * line_render_vertex_shader = "#version 330 core\n" STRINGIFY(

layout(location = 0) in vec3 vert_pos;
layout(location = 3) in vec3 vert_col;

out vec3 frag_vert_col;

uniform mat4 proj_trans;
uniform mat4 view_trans;
uniform mat4 model_trans;

void main()
{
    frag_vert_col = vert_col;
    gl_Position = proj_trans * view_trans * model_trans * vec4(vert_pos, 1.0);
}

);

static const char * line_render_frag_shader = "#version 330 core\n" STRINGIFY(

in vec3 frag_vert_col;
out vec4 frag_color;

void main()
{
    frag_color = vec4(frag_vert_col, 1.0);
}

);

}//end of namespace Physika