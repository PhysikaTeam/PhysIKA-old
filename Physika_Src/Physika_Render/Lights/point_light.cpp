/*
 * @file point_light.cpp
 * @brief class PointLight.
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

#include "point_light.h"
#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"

namespace Physika{

PointLight::PointLight(const Color4f & ambient_col, const Color4f & diffuse_col, const Color4f & specular_col, const Vector3f & pos, float constant_attenuation, float linear_attenuation, float quadratic_attenution)
    :LightBase(ambient_col, diffuse_col, specular_col), pos_(pos), constant_attenuation_(constant_attenuation), linear_attenuation_(linear_attenuation), quadratic_attenution_(quadratic_attenution)
{
    
}

LightType PointLight::type() const
{
    return LightType::POINT_LIGHT;
}

void PointLight::configToCurBindShader(const std::string & light_str)
{
    //call base configToCurBindShader
    this->LightBase::configToCurBindShader(light_str);

    // pos
    std::string pos_str = light_str + ".pos";
    const Vector3f & pos = this->pos();  //note: pos() is override in flash light, so we do not directly use pos_
    openGLSetCurBindShaderVec3(pos_str, pos);

    //constant attenuation 
    std::string constant_atten_str = light_str + ".constant_atten";
    openGLSetCurBindShaderFloat(constant_atten_str, constant_attenuation_);

    //linear attenuation
    std::string linear_atten_str = light_str + ".linear_atten";
    openGLSetCurBindShaderFloat(linear_atten_str, linear_attenuation_);

    //quadratic attenuation
    std::string quadratic_atten_str = light_str + ".quadratic_atten";
    openGLSetCurBindShaderFloat(quadratic_atten_str, quadratic_attenution_);
}

const Vector3f PointLight::pos() const
{
    return this->pos_;
}

void PointLight::setPos(const Vector3f & pos)
{
    this->pos_ = pos;
}

void PointLight::setConstantAttenuation(float constant_atten)
{
    this->constant_attenuation_ = constant_atten;
}

float PointLight::constantAttenuation() const
{
    return this->constant_attenuation_;
}

void PointLight::setLinearAttenuation(float linear_atten)
{
    this->linear_attenuation_ = linear_atten;
}

float PointLight::linearAttenuation() const
{
    return this->linear_attenuation_;
}

void PointLight::setQuadraticAttenuation(float quad_atten)
{
    this->quadratic_attenution_ = quad_atten;
}

float PointLight::quadraticAttenuation() const
{
    return this->quadratic_attenution_;
}

    
}//end of namespace Physika