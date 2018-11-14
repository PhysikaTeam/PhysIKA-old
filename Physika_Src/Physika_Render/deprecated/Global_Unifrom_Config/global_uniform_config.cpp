/*
 * @file global_uniform_config.cpp
 * @Brief global uniform config.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"
#include "global_uniform_config.h"

namespace Physika{

GlobalUniformConfig & GlobalUniformConfig::getSingleton()
{
    static GlobalUniformConfig singleton;
    return singleton;
}

void GlobalUniformConfig::configGlobalUniforms()
{
    //bool & int & float
    for (const auto & ele : bool_uniforms_)
        openGLSetCurBindShaderBool(ele.first, ele.second);
    for (const auto & ele : int_uniforms_)
        openGLSetCurBindShaderInt(ele.first, ele.second);
    for (const auto & ele : float_uniforms_)
        openGLSetCurBindShaderFloat(ele.first, ele.second);

    //vec2 & vec3 & vec4
    for (const auto & ele : vec2_uniforms_)
        openGLSetCurBindShaderVec2(ele.first, ele.second);
    for (const auto & ele : vec3_uniforms_)
        openGLSetCurBindShaderVec3(ele.first, ele.second);
    for (const auto & ele : vec4_uniforms_)
        openGLSetCurBindShaderVec4(ele.first, ele.second);

    //mat2 & mat3 & mat4
    for (const auto & ele : mat2_uniforms_)
        openGLSetCurBindShaderMat2(ele.first, ele.second);
    for (const auto & ele : mat3_uniforms_)
        openGLSetCurBindShaderMat3(ele.first, ele.second);
    for (const auto & ele : mat4_uniforms_)
        openGLSetCurBindShaderMat4(ele.first, ele.second);

    //glm: vec2 & vec3 & vec4
    for (const auto & ele : glm_vec2_uniforms_)
        openGLSetCurBindShaderVec2(ele.first, ele.second);
    for (const auto & ele : glm_vec3_uniforms_)
        openGLSetCurBindShaderVec3(ele.first, ele.second);
    for (const auto & ele : glm_vec4_uniforms_)
        openGLSetCurBindShaderVec4(ele.first, ele.second);

    //glm: mat2 & mat3 & mat4
    for (const auto & ele : glm_mat2_uniforms_)
        openGLSetCurBindShaderMat2(ele.first, ele.second);
    for (const auto & ele : glm_mat3_uniforms_)
        openGLSetCurBindShaderMat3(ele.first, ele.second);
    for (const auto & ele : glm_mat4_uniforms_)
        openGLSetCurBindShaderMat4(ele.first, ele.second);
}

void GlobalUniformConfig::setBool(const std::string & name, bool val)
{
    bool_uniforms_[name] = val;
}

void GlobalUniformConfig::setInt(const std::string & name, int val)
{
    int_uniforms_[name] = val;
}

void GlobalUniformConfig::setFloat(const std::string & name, float val)
{
    float_uniforms_[name] = val;
}

void GlobalUniformConfig::setVec2(const std::string & name, const Vector2f & val)
{
    vec2_uniforms_[name] = val;
}

void GlobalUniformConfig::setVec3(const std::string & name, const Vector3f & val)
{
    vec3_uniforms_[name] = val;
}

void GlobalUniformConfig::setVec4(const std::string & name, const Vector4f & val)
{
    vec4_uniforms_[name] = val;
}

void GlobalUniformConfig::setVec2(const std::string & name, const glm::vec2 & val)
{
    glm_vec2_uniforms_[name] = val;
}

void GlobalUniformConfig::setVec3(const std::string & name, const glm::vec3 & val)
{
    glm_vec3_uniforms_[name] = val;
}

void GlobalUniformConfig::setVec4(const std::string & name, const glm::vec4 & val)
{
    glm_vec4_uniforms_[name] = val;
}

void GlobalUniformConfig::setMat2(const std::string & name, const Matrix2f & val)
{
    mat2_uniforms_[name] = val;
}

void GlobalUniformConfig::setMat3(const std::string & name, const Matrix3f & val)
{
    mat3_uniforms_[name] = val;
}

void GlobalUniformConfig::setMat4(const std::string & name, const Matrix4f & val)
{
    mat4_uniforms_[name] = val;
}

void GlobalUniformConfig::setMat2(const std::string & name, const glm::mat2 & val)
{
    glm_mat2_uniforms_[name] = val;
}

void GlobalUniformConfig::setMat3(const std::string & name, const glm::mat3 & val)
{
    glm_mat3_uniforms_[name] = val;
}

void GlobalUniformConfig::setMat4(const std::string & name, const glm::mat4 & val)
{
    glm_mat4_uniforms_[name] = val;
}

void GlobalUniformConfig::clear()
{
    bool_uniforms_.clear();
    int_uniforms_.clear();
    float_uniforms_.clear();

    vec2_uniforms_.clear();
    vec3_uniforms_.clear();
    vec4_uniforms_.clear();

    mat2_uniforms_.clear();
    mat3_uniforms_.clear();
    mat4_uniforms_.clear();

    glm_vec2_uniforms_.clear();
    glm_vec3_uniforms_.clear();
    glm_vec4_uniforms_.clear();

    glm_mat2_uniforms_.clear();
    glm_mat3_uniforms_.clear();
    glm_mat4_uniforms_.clear();
}
    
}//end of namespace Physika