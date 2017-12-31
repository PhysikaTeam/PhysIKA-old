/*
* @file shader_base.h
* @Brief Class ShaderBase 
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

#include <unordered_map>
#include <string>
#include "shader_program.h"

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Vectors/vector_4d.h"

#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Matrices/matrix_4x4.h"

namespace Physika{

class GLSceneRenderConfig;

class ShaderBase
{
public:
    ShaderBase(GLSceneRenderConfig * render_config);

    //disable copy
    ShaderBase(const ShaderBase & rhs) = delete;
    ShaderBase & operator = (const ShaderBase & rhs) = delete;
    
    virtual ~ShaderBase() = 0;

    void bindAndConfigUniforms();
    void unBind() const;

protected:
    void setBool(const std::string & name, bool val);
    void setInt(const std::string & name, int val);
    void setFloat(const std::string & name, float val);

    void setVec2(const std::string & name, const Vector2f & val);
    void setVec3(const std::string & name, const Vector3f & val);
    void setVec4(const std::string & name, const Vector4f & val);

    void setMat2(const std::string & name, const Matrix2f & val);
    void setMat3(const std::string & name, const Matrix3f & val);
    void setMat4(const std::string & name, const Matrix4f & val);

private:
    void setUniformsFromGLSceneRenderConfig();
    void setCustomUniforms();

private:
    ShaderProgram shader_prog_;
    GLSceneRenderConfig * render_config_;

    //stores all custom uniforms here
    std::unordered_map<std::string, bool> bool_uniforms_;
    std::unordered_map<std::string, int> int_uniforms_;
    std::unordered_map<std::string, float> float_uniforms_;

    std::unordered_map<std::string, Vector2f> vec2_uniforms_;
    std::unordered_map<std::string, Vector3f> vec3_uniforms_;
    std::unordered_map<std::string, Vector4f> vec4_uniforms_;

    std::unordered_map<std::string, Matrix2f> mat2_uniforms_;
    std::unordered_map<std::string, Matrix3f> mat3_uniforms_;
    std::unordered_map<std::string, Matrix4f> mat4_uniforms_;


};
    
}// end of namespace Physika