/*
 * @file flex_spot_light.cpp
 * @brief class FlexSpotLight.
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

#include <glm/gtc/matrix_transform.hpp>
#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"

#include "flex_spot_light.h"


namespace Physika{

FlexSpotLight::FlexSpotLight(const Color4f & ambient_col, const Color4f & diffuse_col, const Color4f & specular_col, Vector3f light_pos, Vector3f light_target, float light_fov, float light_spot_min, float light_spot_max)
    :LightBase(ambient_col, diffuse_col, specular_col), 
     pos_(light_pos), target_(light_target), fov_(light_fov), spot_min_(light_spot_min), spot_max_(light_spot_max)
{
    
}

LightType FlexSpotLight::type() const
{
    return LightType::FLEX_SPOT_LIGHT;
}

void FlexSpotLight::configToCurBindShader(const std::string & light_str)
{
    //call base configToCurBindShader
    this->LightBase::configToCurBindShader(light_str);

    //pos
    std::string pos_str = light_str + ".pos";
    openGLSetCurBindShaderVec3(pos_str, pos_);

    //spot direction
    std::string spot_direction_str = light_str + ".spot_direction";
    Vector3f spot_direction = this->spotDirection();
    openGLSetCurBindShaderVec3(spot_direction_str, spot_direction);

    //spot min
    std::string spot_min_str = light_str + ".spot_min";
    openGLSetCurBindShaderFloat(spot_min_str, spot_min_);

    //spot min
    std::string spot_max_str = light_str + ".spot_max";
    openGLSetCurBindShaderFloat(spot_max_str, spot_max_);

    //light transform
    std::string light_trans_str = light_str + ".light_trans";
    glm::mat4 light_trans = this->lightTransformMatrix();
    openGLSetCurBindShaderMat4(light_trans_str, light_trans);
}

glm::mat4 FlexSpotLight::lightProjMatrix() const
{
    return glm::perspective(glm::radians(this->fov_), 1.0f, 1.0f, 1000.0f);
}

glm::mat4 FlexSpotLight::lightViewMatrix() const
{
    glm::vec3 light_pos = {pos_[0], pos_[1], pos_[2] };
    glm::vec3 light_target = {target_[0], target_[1], target_[2] };

    const Vector3f & light_dir = this->spotDirection();

    glm::vec3 light_up = { 0.0f, 1.0f, 0.0f };
    if (abs(light_dir[0]) < 0.01 && abs(light_dir[2]) < 0.01)
        light_up = { 1.0f, 0.0f, 0.0f };

    return glm::lookAt(light_pos, light_target, light_up);
}

glm::mat4 FlexSpotLight::lightTransformMatrix() const
{
    return lightProjMatrix() * lightViewMatrix();
}

const Vector3f & FlexSpotLight::pos() const
{
    return pos_;
}

const Vector3f & FlexSpotLight::target() const
{
    return target_;
}

Vector3f FlexSpotLight::spotDirection() const
{
    Vector3f light_dir = target_ - pos_;
    return light_dir.normalize();
}

float FlexSpotLight::fov() const
{
    return fov_;
}

float FlexSpotLight::spotMin() const
{
    return spot_min_;
}

float FlexSpotLight::spotMax() const
{
    return spot_max_;
}

void FlexSpotLight::setPos(const Vector3f & light_pos)
{
    pos_ = light_pos;
}

void FlexSpotLight::setTarget(const Vector3f & light_target)
{
    target_ = light_target;
}

void FlexSpotLight::setFov(float light_fov)
{
    fov_ = light_fov;
}

void FlexSpotLight::setSpotMin(float light_spot_min)
{
    spot_min_ = light_spot_min;
}

void FlexSpotLight::setSpotMax(float light_spot_max)
{
    spot_max_ = light_spot_max;
}
    
}//end of namespace Physika