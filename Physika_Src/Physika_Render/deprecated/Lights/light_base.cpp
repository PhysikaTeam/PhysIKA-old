/*
 * @file light_base.cpp 
 * @brief class LightBase.
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

#include <glm/glm.hpp>
#include "Physika_Core/Utilities/glm_utilities.h"
#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"

#include "light_base.h"

namespace Physika{

LightBase::LightBase(const Color4f & ambient_col, const Color4f & diffuse_col, const Color4f & specular_col)
    :ambient_col_(ambient_col), diffuse_col_(diffuse_col), specular_col_(specular_col)
{
    
}

void LightBase::configToCurBindShader(const std::string & light_str)
{
    std::string light_ambient_str = light_str + ".ambient";
    std::string light_diffuse_str = light_str + ".diffuse";
    std::string light_specular_str = light_str + ".specular";

    openGLSetCurBindShaderCol3(light_ambient_str, ambient_col_);
    openGLSetCurBindShaderCol3(light_diffuse_str, diffuse_col_);
    openGLSetCurBindShaderCol3(light_specular_str, specular_col_);
}

const Color4f & LightBase::ambient() const
{
    return ambient_col_;
}

const Color4f & LightBase::diffuse() const
{
    return diffuse_col_;
}

const Color4f & LightBase::specular() const
{
    return specular_col_;
}

void LightBase::setAmbient(const Color4f & ambient)
{
    ambient_col_ = ambient;
}

void LightBase::setDiffuse(const Color4f & diffuse)
{
    diffuse_col_ = diffuse;
}
    
void LightBase::setSpecular(const Color4f & specular)
{
    specular_col_ = specular;
}

void LightBase::enableLighting()
{
    this->enable_lighting_ = true;
}

void LightBase::disableLighting()
{
    this->enable_lighting_ = false;
}

bool LightBase::isEnableLighting() const
{
    return this->enable_lighting_;
}

}//end of namespace Physika