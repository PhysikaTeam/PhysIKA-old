/*
 * @file global_uniform_config.h
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

#pragma once

/*
Note: class GLobalUniformConfig contain all custom global uniforms 
      which will be set for each Shader before executing the draw operation.

    GlobalUniformConfig act as a hook to set opengl glsl uniforms.
*/

#include <string>
#include <unordered_map>

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Vectors/vector_4d.h"

#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Matrices/matrix_4x4.h"

namespace Physika{

class GlobalUniformConfig
{
public:
    static GlobalUniformConfig & getSingleton();
    void configGlobalUniforms();

    //setter
    void setBool(const std::string & name, bool val);
    void setInt(const std::string & name, int val);
    void setFloat(const std::string & name, float val);

    void setVec2(const std::string & name, const Vector2f & val);
    void setVec3(const std::string & name, const Vector3f & val);
    void setVec4(const std::string & name, const Vector4f & val);

    void setVec2(const std::string & name, const glm::vec2 & val);
    void setVec3(const std::string & name, const glm::vec3 & val);
    void setVec4(const std::string & name, const glm::vec4 & val);

    void setMat2(const std::string & name, const Matrix2f & val);
    void setMat3(const std::string & name, const Matrix3f & val);
    void setMat4(const std::string & name, const Matrix4f & val);

    void setMat2(const std::string & name, const glm::mat2 & val);
    void setMat3(const std::string & name, const glm::mat3 & val);
    void setMat4(const std::string & name, const glm::mat4 & val);

    void clear(); //clear all global uniforms

private:
    GlobalUniformConfig() = default;
    GlobalUniformConfig(const GlobalUniformConfig &) = default;
    GlobalUniformConfig & operator = (const GlobalUniformConfig &) = default;

    ~GlobalUniformConfig() = default;

private:
    std::unordered_map<std::string, bool> bool_uniforms_;
    std::unordered_map<std::string, int> int_uniforms_;
    std::unordered_map<std::string, float> float_uniforms_;

    std::unordered_map<std::string, Vector2f> vec2_uniforms_;
    std::unordered_map<std::string, Vector3f> vec3_uniforms_;
    std::unordered_map<std::string, Vector4f> vec4_uniforms_;

    std::unordered_map<std::string, Matrix2f> mat2_uniforms_;
    std::unordered_map<std::string, Matrix3f> mat3_uniforms_;
    std::unordered_map<std::string, Matrix4f> mat4_uniforms_;

    std::unordered_map<std::string, glm::vec2> glm_vec2_uniforms_;
    std::unordered_map<std::string, glm::vec3> glm_vec3_uniforms_;
    std::unordered_map<std::string, glm::vec4> glm_vec4_uniforms_;

    std::unordered_map<std::string, glm::mat2> glm_mat2_uniforms_;
    std::unordered_map<std::string, glm::mat3> glm_mat3_uniforms_;
    std::unordered_map<std::string, glm::mat4> glm_mat4_uniforms_;
};
    
}//end of namespace Physika