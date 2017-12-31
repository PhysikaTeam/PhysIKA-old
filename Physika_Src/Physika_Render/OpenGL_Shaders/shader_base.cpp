/*
* @file shader_base.cpp
* @Brief Class ShaderBase
* @author Wei Chen
*
* This file is part of Physika, a versatile physics simulation library.
* Copyright (C) 2013- Physika Group.
*
* This Source Code Form is subject to the terms of the GNU General Public License v2.0.
* If a copy of the GPL was not distributed with this file, you can obtain one at:
* http://www.gnu.org/licenses/gpl-2.0.html
*/

#include "Physika_Core/Utilities/physika_exception.h"
#include "shader_base.h"

namespace Physika{

ShaderBase::ShaderBase(GLSceneRenderConfig * render_config)
    :render_config_(render_config)
{
    
}

//default dtor
ShaderBase::~ShaderBase() = default;

void ShaderBase::bindAndConfigUniforms()
{
    this->shader_prog_.use();
    this->setUniformsFromGLSceneRenderConfig();
    this->setCustomUniforms();
}

void ShaderBase::unBind() const
{
    this->shader_prog_.unUse();
}

void ShaderBase::setBool(const std::string & name, bool val)
{
    this->bool_uniforms_[name] = val;
}

void ShaderBase::setInt(const std::string & name, int val)
{
    this->int_uniforms_[name] = val;
}

void ShaderBase::setFloat(const std::string & name, float val)
{
    this->float_uniforms_[name] = val;
}

void ShaderBase::setVec2(const std::string & name, const Vector2f & val)
{
    this->vec2_uniforms_[name] = val;
}

void ShaderBase::setVec3(const std::string & name, const Vector3f & val)
{
    this->vec3_uniforms_[name] = val;
}

void ShaderBase::setVec4(const std::string & name, const Vector4f & val)
{
    this->vec4_uniforms_[name] = val;
}

void ShaderBase::setMat2(const std::string & name, const Matrix2f & val)
{
    this->mat2_uniforms_[name] = val;
}

void ShaderBase::setMat3(const std::string & name, const Matrix3f & val)
{
    this->mat3_uniforms_[name] = val;
}

void ShaderBase::setMat4(const std::string & name, const Matrix4f & val)
{
    this->mat4_uniforms_[name] = val;
}

void ShaderBase::setUniformsFromGLSceneRenderConfig()
{
    //to do
    throw PhysikaException("error: not implemented!");
}

void ShaderBase::setCustomUniforms()
{
    //set for bool, int, float
    for (const auto & ele : bool_uniforms_)
        this->shader_prog_.setBool(ele.first, ele.second);
    for (const auto & ele : int_uniforms_)
        this->shader_prog_.setInt(ele.first, ele.second);
    for (const auto & ele : float_uniforms_)
        this->shader_prog_.setFloat(ele.first, ele.second);

    //set for vec2, vec3, vec4
    for (const auto & ele : vec2_uniforms_)
        this->shader_prog_.setVec2(ele.first, ele.second);
    for (const auto & ele : vec3_uniforms_)
        this->shader_prog_.setVec3(ele.first, ele.second);
    for (const auto & ele : vec4_uniforms_)
        this->shader_prog_.setVec4(ele.first, ele.second);

    //set for mat2, mat3, mat4
    for (const auto & ele : mat2_uniforms_)
        this->shader_prog_.setMat2(ele.first, ele.second);
    for (const auto & ele : mat3_uniforms_)
        this->shader_prog_.setMat3(ele.first, ele.second);
    for (const auto & ele : mat4_uniforms_)
        this->shader_prog_.setMat4(ele.first, ele.second);

}

}//end of namespace Physika