/*
* @file plane_shader_srcs.h
* @Brief GLSL shaders for plane
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

static const char * plane_vertex_shader = "#version 330 core\n" STRINGIFY(

layout(location = 0) in vec3 vert_pos;
layout(location = 1) in vec3 vert_normal;

out vec3 frag_pos;
out vec3 frag_normal;

uniform mat4 proj_trans;
uniform mat4 view_trans;
uniform mat4 model_trans;

//-------------------------------------------------------------------------------------------------

void main()
{
    frag_pos = (model_trans * vec4(vert_pos, 1.0)).xyz;
    frag_normal = mat3(transpose(inverse(model_trans))) * vert_normal;
    gl_Position = proj_trans * view_trans * vec4(frag_pos, 1.0);
}

);

//-------------------------------------------------------------------------------------------------

static const char * plane_frag_shader = "#version 330 core\n" STRINGIFY(

in vec3 frag_pos;
in vec3 frag_normal;

out vec4 frag_color;

//-------------------------------------------------------------------------------------------------

uniform vec3 view_pos;
uniform bool use_light = true;
uniform bool use_shadow_map;
uniform bool render_grid;

//------------------------------------------------------------------------------------------------------------

struct Material
{
    vec3 Ka;
    vec3 Kd;
    vec3 Ks;
    float shininess;
    float alpha;
};

vec3 default_Ka = vec3(0.2);
vec3 default_Kd = vec3(0.8);
vec3 default_Ks = vec3(0.0);
float default_shininess = 0.0;

uniform bool use_material;
uniform Material material;

vec3 Ka()
{
    return use_material ? material.Ka : default_Ka;
}

vec3 Kd()
{
    return use_material ? material.Kd : default_Kd;
}

vec3 Ks()
{
    return use_material ? material.Ks : default_Ks;
}

float shininess()
{
    return use_material ? material.shininess : default_shininess;
}

//------------------------------------------------------------------------------------------------------------

struct DirectionalLight
{
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    vec3 direction;
};

uniform int directional_light_num = 0;
uniform DirectionalLight directional_lights[5];

vec3 calcuDirectionalLightColor()
{
    vec3 total_light_color = vec3(0.0);
    for (int i = 0; i < directional_light_num; ++i)
    {
        DirectionalLight light = directional_lights[i];

        //ambient
        vec3 ambient = light.ambient * Ka();

        //diffuse
        vec3 normal = normalize(frag_normal);
        vec3 light_dir = normalize(-light.direction);
        float diff_factor = max(dot(normal, light_dir), 0.0);
        vec3 diffuse = diff_factor * light.diffuse * Kd();

        //specular
        vec3 view_dir = normalize(view_pos - frag_pos);
        vec3 halfway_dir = normalize(light_dir + view_dir);
        float spec_factor = pow(max(dot(normal, halfway_dir), 0.0), shininess());
        vec3 specular = spec_factor * light.specular * Ks();

        total_light_color += ambient + diffuse + specular;
    }
    return total_light_color;
}

//------------------------------------------------------------------------------------------------------------

struct PointLight
{
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    vec3 pos;
    float constant_atten;
    float linear_atten;
    float quadratic_atten;
};

uniform int point_light_num = 0;
uniform PointLight point_lights[5];

vec3 calcuPointLightColor()
{
    vec3 total_light_color = vec3(0.0);
    for (int i = 0; i < point_light_num; ++i)
    {
        PointLight light = point_lights[i];

        //ambient
        vec3 ambient = light.ambient * Ka();

        //diffuse
        vec3 normal = normalize(frag_normal);
        vec3 light_dir = normalize(light.pos - frag_pos);
        float diff_factor = max(dot(normal, light_dir), 0.0);
        vec3 diffuse = diff_factor * light.diffuse * Kd();

        //specular
        vec3 view_dir = normalize(view_pos - frag_pos);
        vec3 halfway_dir = normalize(light_dir + view_dir);
        float spec_factor = pow(max(dot(normal, halfway_dir), 0.0), shininess());
        vec3 specular = spec_factor * light.specular * Ks();

        //attenuation
        float distance = length(light.pos - frag_pos);
        float attenuation = 1.0 / (light.constant_atten + light.linear_atten * distance + light.quadratic_atten * distance * distance);

        total_light_color += ambient + (diffuse + specular) * attenuation;
    }
    return total_light_color;
}

//------------------------------------------------------------------------------------------------------------

struct SpotLight
{
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    vec3 pos;
    float constant_atten;
    float linear_atten;
    float quadratic_atten;

    vec3 spot_direction;
    float spot_exponent;
    float spot_cutoff;       //in radians

    bool use_spot_outer_cutoff;
    float spot_outer_cutoff; //in radians

    mat4 light_trans;
};

struct SpotLightShadowMap
{
    bool has_shadow_map;
    sampler2D shadow_map_tex;
};

uniform int spot_light_num = 0;
uniform SpotLight spot_lights[10];
uniform SpotLightShadowMap spot_light_shadow_maps[10];

float calcuSpotLightShadowAttenuation(int light_id)
{
    vec4 frag_light_space_pos = spot_lights[light_id].light_trans * vec4(frag_pos, 1.0);
    //vec4 frag_light_space_pos = frag_spot_light_space_pos[light_id];
    vec3 pos = frag_light_space_pos.xyz / frag_light_space_pos.w;

    //convert to texture coord space
    vec3 uvw = pos * 0.5 + 0.5;

    //clip
    if (uvw.x  < 0.0 || uvw.x > 1.0)
        return 1.0;
    if (uvw.y < 0.0 || uvw.y > 1.0)
        return 1.0;

    vec2 shadow_taps[12];
    shadow_taps[0] = vec2(-0.326212f, -0.40581f);
    shadow_taps[1] = vec2(0.840144f, -0.07358f);
    shadow_taps[2] = vec2(-0.695914f, 0.457137f);
    shadow_taps[3] = vec2(-0.203345f, 0.620716f);
    shadow_taps[4] = vec2(0.96234f, -0.194983f);
    shadow_taps[5] = vec2(0.473434f, -0.480026f);
    shadow_taps[6] = vec2(0.519456f, 0.767022f);
    shadow_taps[7] = vec2(0.185461f, -0.893124f);
    shadow_taps[8] = vec2(0.507431f, 0.064425f);
    shadow_taps[9] = vec2(0.89642f, 0.412458f);
    shadow_taps[10] = vec2(-0.32194f, -0.932615f);
    shadow_taps[11] = vec2(-0.791559f, -0.59771f);

    float shadow_atten = 0.0f;

    float radius = 0.002f;
    float bias = 0.000f;
    const int num_taps = 12;
    for (int i = 0; i < num_taps; ++i)
    {
        bool is_shadow = uvw.z - bias > texture(spot_light_shadow_maps[light_id].shadow_map_tex, uvw.xy + shadow_taps[i] * radius).r;
        if (is_shadow == false) shadow_atten += 1;
    }
    shadow_atten /= num_taps;

    return shadow_atten;
}

vec3 calcuSpotLightColor()
{
    vec3 total_light_color = vec3(0.0);
    for (int i = 0; i < spot_light_num; ++i)
    {
        SpotLight light = spot_lights[i];

        vec3 light_dir = normalize(light.pos - frag_pos);
        float theta = dot(light_dir, normalize(-light.spot_direction));
        float phi = cos(light.spot_cutoff);
        float gamma = cos(light.spot_outer_cutoff);

        if (light.use_spot_outer_cutoff == false && theta <= phi)
            continue;

        //ambient
        vec3 ambient = light.ambient * Ka();

        //diffuse
        vec3 normal = normalize(frag_normal);
        float diff_factor = max(dot(normal, light_dir), 0.0);
        vec3 diffuse = diff_factor * light.diffuse * Kd();

        //specular
        vec3 view_dir = normalize(view_pos - frag_pos);
        vec3 halfway_dir = normalize(light_dir + view_dir);
        float spec_factor = pow(max(dot(normal, halfway_dir), 0.0), shininess());
        vec3 specular = spec_factor * light.specular * Ks();

        //attenuation
        float distance = length(light.pos - frag_pos);
        float attenuation = 1.0 / (light.constant_atten + light.linear_atten * distance + light.quadratic_atten * distance * distance);

        //spot factor
        float spot_factor = pow(theta, light.spot_exponent);

        //intensity 
        float intensity = 1.0;
        if (light.use_spot_outer_cutoff == true)
            intensity = clamp((theta - gamma) / (phi - gamma), 0.0, 1.0);

        vec3 light_color = ambient + (diffuse + specular) * attenuation * spot_factor * intensity;

        if (use_shadow_map && spot_light_shadow_maps[i].has_shadow_map)
            light_color *= calcuSpotLightShadowAttenuation(i);

        //vec3 light_color = vec3(calcuSpotLightShadowAttenuation(i));

        total_light_color += light_color;
    }
    return total_light_color;
}

//------------------------------------------------------------------------------------------------------------

struct FlexSpotLight
{
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    vec3 pos;

    vec3 spot_direction;
    float spot_min;
    float spot_max;

    mat4 light_trans;
};

struct FlexSpotLightShadowMap
{
    bool has_shadow_map;
    sampler2D shadow_map_tex;
};

uniform int flex_spot_light_num = 0;
uniform FlexSpotLight flex_spot_lights[10];
uniform FlexSpotLightShadowMap flex_spot_light_shadow_maps[10];

float calcuFlexSpotLightShadowAttenuation(int light_id)
{
    vec4 frag_light_space_pos = flex_spot_lights[light_id].light_trans * vec4(frag_pos, 1.0);
    //vec4 frag_light_space_pos = frag_flex_spot_light_space_pos[light_id];
    vec3 pos = frag_light_space_pos.xyz / frag_light_space_pos.w;

    //convert to texture coord space
    vec3 uvw = pos * 0.5 + 0.5;

    //clip
    if (uvw.x  < 0.0 || uvw.x > 1.0)
        return 1.0;
    if (uvw.y < 0.0 || uvw.y > 1.0)
        return 1.0;

    vec2 shadow_taps[12];
    shadow_taps[0] = vec2(-0.326212f, -0.40581f);
    shadow_taps[1] = vec2(0.840144f, -0.07358f);
    shadow_taps[2] = vec2(-0.695914f, 0.457137f);
    shadow_taps[3] = vec2(-0.203345f, 0.620716f);
    shadow_taps[4] = vec2(0.96234f, -0.194983f);
    shadow_taps[5] = vec2(0.473434f, -0.480026f);
    shadow_taps[6] = vec2(0.519456f, 0.767022f);
    shadow_taps[7] = vec2(0.185461f, -0.893124f);
    shadow_taps[8] = vec2(0.507431f, 0.064425f);
    shadow_taps[9] = vec2(0.89642f, 0.412458f);
    shadow_taps[10] = vec2(-0.32194f, -0.932615f);
    shadow_taps[11] = vec2(-0.791559f, -0.59771f);

    float shadow_atten = 0.0f;

    float radius = 0.002f;
    float bias = 0.000f;
    const int num_taps = 12;
    for (int i = 0; i < num_taps; ++i)
    {
        bool is_shadow = uvw.z - bias > texture(flex_spot_light_shadow_maps[light_id].shadow_map_tex, uvw.xy + shadow_taps[i] * radius).r;
        if (is_shadow == false) shadow_atten += 1;
    }
    shadow_atten /= num_taps;

    return shadow_atten;
}

vec3 calcuFlexSpotLightColor()
{
    vec3 total_light_color = vec3(0.0);
    for (int i = 0; i < flex_spot_light_num; ++i)
    {
        FlexSpotLight light = flex_spot_lights[i];

        //light attenuation
        vec3 light_to_frag_pos_dir = normalize(frag_pos - light.pos);

        //vec4 frag_light_space_pos = light.light_trans * vec4(frag_pos, 1.0);
        //vec3 pos = frag_light_space_pos.xyz / frag_light_space_pos.w;
        //float dot_val = dot(light_to_frag_pos_dir.xy, pos.xy);

        float dot_val = dot(light_to_frag_pos_dir, light.spot_direction);

        //float light_atten = light.spot_min + dot_val * (light.spot_max - light.spot_min);
        float light_atten = smoothstep(light.spot_min, light.spot_max, dot_val);
        light_atten = max(light_atten, 0.05);

        //diffuse
        float wrap = 0.0;
        float diff_factor = max(0.0, (dot(-light.spot_direction, frag_normal) + wrap) / (1.0 + wrap));
        vec3 diffuse = diff_factor * light_atten * Kd();

        //ambient
        vec3 bright = vec3(0.03, 0.025, 0.025)*1.5;
        vec3 dark = vec3(0.025, 0.025, 0.03);
        vec3 ambient = 4.0 * light_atten * Ka() * mix(dark, bright, dot(-light.spot_direction, frag_normal) * 0.5 + 0.5);

        vec3 light_color = diffuse + ambient;

        if (use_shadow_map && flex_spot_light_shadow_maps[i].has_shadow_map)
            light_color *= calcuFlexSpotLightShadowAttenuation(i);

        //light_color = vec3(light.spot_max);
        //light_color = vec3(dot_val);
        //light_color = vec3(light_atten);

        total_light_color += light_color;
    }
    return total_light_color;
}

//----------------------------------------------------------------------------------------

float filterwidth(vec2 v)
{
    vec2 fw = max(abs(dFdx(v)), abs(dFdy(v)));
    return max(fw.x, fw.y);
}

vec2 bump(vec2 x)
{
    return (floor(0.5 * x) + 2.0 * max(0.5 * x - floor(0.5 * x) - 0.5, 0));
}

float checker(vec2 uv)
{
    float width = filterwidth(uv);
    vec2 p0 = uv - 0.5 * width;
    vec2 p1 = uv + 0.5 * width;

    vec2 i = (bump(p1) - bump(p0)) / width;
    return i.x * i.y + (1 - i.x) * (1 - i.y);
}

//----------------------------------------------------------------------------------------

void main()
{
    vec3 total_light_color = vec3(0.0);
    total_light_color += calcuDirectionalLightColor();
    total_light_color += calcuPointLightColor();
    total_light_color += calcuSpotLightColor();
    total_light_color += calcuFlexSpotLightColor();

    //checker
    if (render_grid)
    {
        if (frag_normal.y > 0.995)
            total_light_color *= 1.0 - 0.25 * checker(vec2(frag_pos.x, frag_pos.z));
        else if (abs(frag_normal.z) > 0.995)
            total_light_color *= 1.0 - 0.25 * checker(vec2(frag_pos.y, frag_pos.x));
    }

    frag_color = vec4(total_light_color, 1.0);
}

);

}//end of namespace Physika