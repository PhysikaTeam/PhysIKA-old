/*
* @file shader_program.h
* @Brief Class ShaderProgram used to generate openGL program from shaders
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

#ifndef PHYSIKA_RENDER_OPENGL_SHADERS_SHADER_PROGRAM_H
#define PHYSIKA_RENDER_OPENGL_SHADERS_SHADER_PROGRAM_H

#include <string>
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"

namespace Physika {

class ShaderProgram
{
public:

    ShaderProgram() = default;

    ~ShaderProgram();
    /*
    ShaderProgram(const char * vertex_shader_source,
                  const char * fragment_shader_source,
                  const char * geometry_shader_source = nullptr,
                  const char * tess_control_shader_source = nullptr,
                  const char * tess_evaluation_shader_source = nullptr);
    */

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

    void destory();

    ShaderProgram(const ShaderProgram &) = delete;
    ShaderProgram & operator = (const ShaderProgram &) = delete;

    void use() const;
    void unUse() const;

    bool isValid() const;
    GLuint id() const;


private:
    GLuint program_ = 0;
};

} // end of namespace Physika

#endif //PHYSIKA_RENDER_OPENGL_SHADERS_SHADER_PROGRAM_H
