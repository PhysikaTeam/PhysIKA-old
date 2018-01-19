/*
* @file fluid_point_shader_srcs.h
* @Brief GLSL shaders for fluid point render task
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

namespace Physika{

#define STRINGIFY(A) #A

static const char * fluid_point_vert_shader = "#version 330 core\n" STRINGIFY(

layout(location = 0) in vec4 vert_pos;

layout(location = 5) in float density;
layout(location = 6) in int phase;

uniform mat4 proj_trans;
uniform mat4 view_trans;
uniform mat4 model_trans;

uniform float pointRadius;  
uniform float pointScale;   

uniform mat4 lightTransform;
uniform vec3 lightDir;

uniform vec4 colors[8];
uniform int mode;

out vec3 frag_pos;
out vec4 frag_light_space_pos;
out vec4 frag_view_space_light_dir;
out vec4 frag_view_space_pos;
out vec4 reflect_col;

void main()
{

    vec4 view_pos = view_trans * model_trans * vec4(vert_pos.xyz, 1.0);

    gl_Position = proj_trans * view_trans * model_trans  * vec4(vert_pos.xyz, 1.0);
    gl_PointSize = -1.0 * pointScale * (pointRadius / view_pos.z);

    frag_pos.xyz = vert_pos.xyz;
    frag_view_space_pos.xyz = view_pos.xyz;
    frag_light_space_pos = lightTransform*vec4(vert_pos.xyz - lightDir*pointRadius*2.0, 1.0);
    frag_view_space_light_dir = view_trans * model_trans * vec4(lightDir, 0.0);

    if (mode == 1)
    {
        // density visualization
        if (density < 0.0f)
            reflect_col.xyz = mix(vec3(0.1, 0.1, 1.0), vec3(0.1, 1.0, 1.0), -density);
        else
            reflect_col.xyz = mix(vec3(1.0, 1.0, 1.0), vec3(0.1, 0.2, 1.0), density);
    }
    else if (mode == 2)
    {
        gl_PointSize *= clamp(vert_pos.w*0.25, 0.0f, 1.0);
        reflect_col.xyzw = vec4(clamp(vert_pos.w*0.05, 0.0f, 1.0));
    }
    else
    {
        reflect_col.xyz = mix(colors[phase % 8].xyz*2.0, vec3(1.0), 0.1);
    }
}

);

///////////////////////////////////////////////////////////////////////////////////////////////////////

static const char * fluid_point_frag_shader = "#version 330 core\n" STRINGIFY(

in vec3 frag_pos;
in vec4 frag_light_space_pos;
in vec4 frag_view_space_light_dir;
in vec4 frag_view_space_pos;
in vec4 reflect_col;

out vec4 frag_color;

uniform mat4 proj_trans;

uniform vec3 lightDir;
uniform vec3 lightPos;
uniform float spotMin;
uniform float spotMax;
uniform int mode;

uniform sampler2D shadowTex;
uniform vec2 shadowTaps[12];

uniform float pointRadius;  

// sample shadow map
float shadowSample()
{
    vec3 pos = vec3(frag_light_space_pos.xyz / frag_light_space_pos.w);
    vec3 uvw = (pos.xyz*0.5) + vec3(0.5);

    // user clip
    if (uvw.x  < 0.0 || uvw.x > 1.0)
        return 1.0;
    if (uvw.y < 0.0 || uvw.y > 1.0)
        return 1.0;

    float s = 0.0;
    float radius = 0.002;

    for (int i = 0; i < 8; i++)
    {
        bool is_shadow = uvw.z > texture(shadowTex, uvw.xy + shadowTaps[i] * radius).r;
        if (is_shadow == false) s += 1;
    }

    s /= 8.0;
    return s;
}

float square(float x) { return x*x; }

void main()
{
    // calculate normal from texture coordinates
    vec3 normal;
    normal.xy = gl_PointCoord.xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(normal.xy, normal.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
    normal.z = sqrt(1.0 - mag);

    if (mode == 2)
    {
        float alpha = normal.z*reflect_col.w;
        frag_color.xyz = reflect_col.xyz*alpha;
        frag_color.w = alpha;
        return;
    }

    // calculate lighting
    //float shadow = shadowSample();
    float shadow = 1.0;

    vec3 lVec = normalize(frag_pos - lightPos);
    vec3 lPos = vec3(frag_light_space_pos.xyz / frag_light_space_pos.w);
    float attenuation = max(smoothstep(spotMax, spotMin, dot(lPos.xy, lPos.xy)), 0.05);

    vec3 diffuse = vec3(0.9, 0.9, 0.9);
    vec3 reflectance = reflect_col.xyz;

    vec3 Lo = diffuse*reflectance*max(0.0, square(-dot(frag_view_space_light_dir.xyz, normal)*0.5 + 0.5))*max(0.2, shadow)*attenuation;

    frag_color = vec4(pow(Lo, vec3(1.0 / 2.2)), 1.0);

    vec3 eye_pos = frag_view_space_pos.xyz + normal*pointRadius;
    vec4 ndc_pos = proj_trans * vec4(eye_pos, 1.0);
    ndc_pos.z /= ndc_pos.w;
    gl_FragDepth = ndc_pos.z*0.5 + 0.5;
}

);

    
}//end of namespace Physika