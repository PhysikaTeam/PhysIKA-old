/*
* @file screen_based_shader_srcs.h
* @Brief GLSL shaders for screen based render manager
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

static const char * screen_based_vertex_shader = "#version 330 core\n" STRINGIFY(

layout(location = 0) in vec3 vert_pos;
layout(location = 2) in vec2 vert_tex_coord;

out vec2 frag_tex_coord;

void main()
{
    frag_tex_coord = vert_tex_coord;
    gl_Position = vec4(vert_pos, 1.0);
}

);

static const char * screen_based_frag_shader = "#version 330 core\n" STRINGIFY(

in vec2 frag_tex_coord;
out vec4 frag_color;

uniform sampler2D tex;
uniform bool use_gamma_correction;
uniform bool use_hdr;

void main()
{
    vec3 light_color = texture(tex, frag_tex_coord).rgb;

    if (use_gamma_correction)
        light_color = pow(light_color, vec3(1.0 / 2.2));

    if (use_hdr)
        light_color = light_color / (vec3(1.0) + light_color);

    frag_color = vec4(light_color, 1.0);
}

);

}//end of namespace Physika