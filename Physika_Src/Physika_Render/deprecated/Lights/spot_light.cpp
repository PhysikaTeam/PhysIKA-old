/*
 * @file spot_light.cpp 
 * @Brief a light class for OpenGL.
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
#include "spot_light.h"
#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"

namespace Physika{

SpotLight::SpotLight(const Color4f & ambient_col, const Color4f & diffuse_col, const Color4f & specular_col, const Vector3f & pos, float constant_attenuation, float linear_attenuation, float quadratic_attenution, const Vector3f & spot_direction, float spot_exponent, float spot_cutoff)
    :PointLight(ambient_col, diffuse_col, specular_col, pos, constant_attenuation, linear_attenuation, quadratic_attenution),
     spot_direction_(spot_direction), spot_exponent_(spot_exponent), spot_cutoff_(spot_cutoff)
{
    
}

LightType SpotLight::type() const
{
    return LightType::SPOT_LIGHT;
}

void SpotLight::configToCurBindShader(const std::string & light_str)
{
    //call base configToCurBindShader
    this->PointLight::configToCurBindShader(light_str);

    //spot direction
    std::string spot_direction_str = light_str + ".spot_direction";
    Vector3f spot_direction = this->spotDirection(); //note: spotDirection() is override in flash light, so we do not directly use spot_direction_
    openGLSetCurBindShaderVec3(spot_direction_str, spot_direction);

    //spot exponent
    std::string spot_exponent_str = light_str + ".spot_exponent";
    openGLSetCurBindShaderFloat(spot_exponent_str, spot_exponent_);

    //spot cutoff
    std::string spot_cutoff_str = light_str + ".spot_cutoff";
    openGLSetCurBindShaderFloat(spot_cutoff_str, glm::radians(spot_cutoff_));

    //spot outer cutoff
    std::string use_spot_outer_cutoff_str = light_str + ".use_spot_outer_cutoff";
    if (spot_outer_cutoff_ < 1e-3)
    {
        openGLSetCurBindShaderBool(use_spot_outer_cutoff_str, false);
    }
    else
    {
        openGLSetCurBindShaderBool(use_spot_outer_cutoff_str, true);
        std::string spot_outer_cutoff_str = light_str + ".spot_outer_cutoff";
        openGLSetCurBindShaderFloat(spot_outer_cutoff_str, glm::radians(spot_outer_cutoff_));
    }

    //light transform
    std::string light_trans_str = light_str + ".light_trans";
    glm::mat4 light_trans = this->lightTransformMatrix();
    openGLSetCurBindShaderMat4(light_trans_str, light_trans);
}

glm::mat4 SpotLight::lightProjMatrix() const
{
    //need further consideration
    return glm::perspective(glm::radians(45.0f), 1.0f, 1.0f, 1000.0f);
}

glm::mat4 SpotLight::lightViewMatrix() const
{
    const Vector3f & light_pos = this->pos();
    const Vector3f & light_dir = this->spotDirection();

    Vector3f light_target = light_pos + light_dir;

    glm::vec3 glm_light_pos = { light_pos[0], light_pos[1], light_pos[2] };
    glm::vec3 glm_light_target = { light_target[0], light_target[1], light_target[2] };

    glm::vec3 glm_light_up = { 0.0f, 1.0f, 0.0f };
    if (abs(light_dir[0]) < 0.01 && abs(light_dir[2]) < 0.01)
        glm_light_up = { 1.0f, 0.0f, 0.0f };

    return glm::lookAt(glm_light_pos, glm_light_target, glm_light_up);
}

glm::mat4 SpotLight::lightTransformMatrix() const
{
    return this->lightProjMatrix() * this->lightViewMatrix();
}

void SpotLight::setSpotDirection(const Vector3f & direction)
{
    this->spot_direction_ = direction;
}

const Vector3f SpotLight::spotDirection() const
{
    return this->spot_direction_;
}

void SpotLight::setSpotExponent(float exponent)
{
    this->spot_exponent_ = exponent;
}

float SpotLight::spotExponent() const
{
    return this->spot_exponent_;
}

void SpotLight::setSpotCutoff(float cutoff)
{
    this->spot_cutoff_ = cutoff;
}

float SpotLight::spotCutoff() const
{
    return this->spot_cutoff_;
}

void SpotLight::setSpotOuterCutoff(float outer_cutoff)
{
    this->spot_outer_cutoff_ = outer_cutoff;
}

float SpotLight::spotOuterCutoff() const
{
    return this->spot_outer_cutoff_;
}

}// end of namespace Physika
