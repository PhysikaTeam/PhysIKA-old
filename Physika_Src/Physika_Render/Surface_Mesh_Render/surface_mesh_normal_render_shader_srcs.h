/*
* @file surface_mesh_normal_render_shader_srcs.h
* @Brief GLSL shaders for normal render of surface mesh
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static const char * surface_mesh_normal_render_vertex_shader = "#version 330 core\n" STRINGIFY(

layout(location = 0) in vec3 vert_pos;
layout(location = 1) in vec3 vert_normal;

out vec3 frag_normal;

uniform mat4 proj_trans;
uniform mat4 view_trans;
uniform mat4 model_trans;

void main()
{
    frag_normal = mat3(transpose(inverse(model_trans))) * vert_normal;
    gl_Position = proj_trans * view_trans * model_trans * vec4(vert_pos, 1.0);
}

);


static const char * surface_mesh_normal_render_frag_shader = "#version 330 core\n" STRINGIFY(

in vec3 frag_normal;
out vec4 frag_color;

uniform bool map_to_color_space;

void main()
{
    if (map_to_color_space)
        frag_color = vec4(0.5 + 0.5 *normalize(frag_normal) / 2.0, 1.0);
    else
        frag_color = vec4(normalize(frag_normal), 1.0);
}

);
    
}//end of namespace Physika