/*
* @file shader.h
* @Brief Class Shader 
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

#include "shader_program.h"

namespace Physika{

class Shader
{
public:
    Shader() = default;

    //disable copy
    Shader(const Shader & rhs) = delete;
    Shader & operator = (const Shader & rhs) = delete;
    
    virtual ~Shader() = default;

    void createFromCStyleString(const char * vertex_shader_source,
                                const char * fragment_shader_source,
                                const char * geometry_shader_source = nullptr,
                                const char * tess_control_shader_source = nullptr,
                                const char * tess_evaluation_shader_source = nullptr);

    void  createFromFile(const std::string & vertex_shader_file,
                         const std::string & fragment_shader_file,
                         const std::string & geometry_shader_file = {},
                         const std::string & tess_control_shader_file = {},
                         const std::string & tess_evaluation_shader_file = {});

    void createFromString(const std::string & vertex_shader_str,
                          const std::string & fragment_shader_str,
                          const std::string & geometry_shader_str = {},
                          const std::string & tess_control_shader_str = {},
                          const std::string & tess_evaluation_shader_str = {});

    void bindAndConfigBasicUniforms();
    void bind();
    void configCameraUniforms();
    void configLightUniforms();
    void unBind() const;

private:
    ShaderProgram shader_prog_;
};
    
}// end of namespace Physika