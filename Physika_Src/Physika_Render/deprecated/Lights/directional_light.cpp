/*
 * @file directioan_light.cpp
 * @brief class DirectionalLight.
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

#include "directional_light.h"
#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"

namespace Physika{

DirectionalLight::DirectionalLight(const Color4f & ambient_col, const Color4f & diffuse_col, const Color4f & specular_col, const Vector3f & direction)
    :LightBase(ambient_col, diffuse_col, specular_col), direction_(direction)
{
    
}

LightType DirectionalLight::type() const
{
    return LightType::DIRECTIONAL_LIGHT;
}

void DirectionalLight::configToCurBindShader(const std::string & light_str)
{
    //call base configToCurBindShader
    this->LightBase::configToCurBindShader(light_str);

    //direction
    std::string direction_str = light_str + ".direction";
    openGLSetCurBindShaderVec3(direction_str, direction_);

}

const Vector3f & DirectionalLight::direction() const
{
    return this->direction_;
}

void DirectionalLight::setDirection(const Vector3f & direction)
{
    this->direction_ = direction;
}
    
}
