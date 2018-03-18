/*
 * @file shader_convention.h 
 * @Brief shader convention
 * @author: Wei Chen
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

/*
本文档将描述关于shader中关于数据结构和布局的相应公约，编码者需严格遵守此公约，以正确向shader传递数据。

------------------------------------------------------------------------------------------------------
顶点属性：

layout (location = 0) in vec3 vert_pos;          //顶点位置
layout (location = 1) in vec3 vert_normal;       //顶点法向
layout (location = 2) in vec2 vert_tex_coord;    //顶点纹理坐标
layout (location = 3) in vec3 vert_col;          //顶点颜色 
layout (location = 4) in vec3 vert_vector;       //顶点自定义向量

layout (location = 5) in float density;          //流体密度
layout (location = 6) in int phase;              //流体phase

------------------------------------------------------------------------------------------------------
照相机：

投影矩阵：
uniform mat4 proj_trans;

视图矩阵:
uniform mat4 view_trans;

模型矩阵:
uniform mat4 model_trans;

观察位置:
uniform vec3 view_pos;

------------------------------------------------------------------------------------------------------

光照和阴影：

struct DirectionalLight
{
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    
    vec3 direction;
};

uniform int directional_light_num = 0;
uniform DirectionalLight directional_lights[5];

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

uniform bool use_light;
uniform bool use_shadow_map; //全局设置是否使用shadow_map

------------------------------------------------------------------------------------------------------

材质：

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

------------------------------------------------------------------------------------------------------

纹理：

uniform bool use_tex;
uniform bool has_tex;
uniform sampler2D tex; //texture unit 0

------------------------------------------------------------------------------------------------------

颜色：

uniform bool use_custom_col;  //对应顶点输入 layout (location = 3) in vec3 vert_col; 


*/