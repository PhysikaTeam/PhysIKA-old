/*
* @file shader_program.cpp
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

#include <iostream>
#include <fstream>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Physika_Core/Utilities/physika_exception.h"

//glew.h must be included before gl.h
#include <GL/glew.h>
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"

#include "Physika_Render/OpenGL_Shaders/shader_program.h"

namespace Physika {

/*
 * glslPrintShaderLog, output error message if fail to compile shader sources
 */

void glslPrintShaderLog(GLuint obj)
{
    int infologLength = 0;
    int charsWritten = 0;
    char *infoLog;

    GLint result;
    glGetShaderiv(obj, GL_COMPILE_STATUS, &result);

    // only print log if compile fails
    if (result == GL_FALSE)
    {
        glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &infologLength);

        if (infologLength > 1)
        {
            infoLog = (char *)malloc(infologLength);
            glGetShaderInfoLog(obj, infologLength, &charsWritten, infoLog);
            printf("%s\n", infoLog);
            free(infoLog);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
ShaderProgram::ShaderProgram(const char * vertex_shader_source,
                             const char * fragment_shader_source,
                             const char * geometry_shader_source,
                             const char * tess_control_shader_source,
                             const char * tess_evaluation_shader_source)
{
    this->createFromCStyleString(vertex_shader_source, 
                                 fragment_shader_source, 
                                 geometry_shader_source, 
                                 tess_control_shader_source,
                                 tess_evaluation_shader_source);
}
*/

ShaderProgram::~ShaderProgram()
{
    this->destory();
}

void ShaderProgram::createFromCStyleString(const char * vertex_shader_source,
                                           const char * fragment_shader_source,
                                           const char * geometry_shader_source,
                                           const char * tess_control_shader_source,
                                           const char * tess_evaluation_shader_source)
{
    //destroy before create
    this->destory();

    //create shader program
    this->program_ = glCreateProgram();


    GLuint shaders[5] = { 0, 0, 0, 0, 0 };

    GLuint types[5] = {
                        GL_VERTEX_SHADER,
                        GL_FRAGMENT_SHADER,
                        GL_GEOMETRY_SHADER,
                        GL_TESS_CONTROL_SHADER,
                        GL_TESS_CONTROL_SHADER
                      };

    const char * sources[5] = {
                                vertex_shader_source,
                                fragment_shader_source,
                                geometry_shader_source,
                                tess_control_shader_source,
                                tess_evaluation_shader_source
                              };

    for (int i = 0; i < 5; ++i)
    {
        if (sources[i] != NULL)
        {
            shaders[i] = glCreateShader(types[i]);
            glShaderSource(shaders[i], 1, &sources[i], 0);
            glCompileShader(shaders[i]);

            //output error message if fails
            glslPrintShaderLog(shaders[i]);

            glAttachShader(program_, shaders[i]);
        }
    }

    glLinkProgram(program_);

    // check if program linked
    GLint success = 0;
    glGetProgramiv(this->program_, GL_LINK_STATUS, &success);
    if (!success)
    {
        char temp[256];
        glGetProgramInfoLog(this->program_, 256, 0, temp);

        std::cerr << "Failed to link program:" << std::endl;
        std::cerr << temp << std::endl;

        glDeleteProgram(this->program_);
        program_ = 0;

        std::exit(EXIT_FAILURE);
    }

    for (int i = 0; i < 5; ++i)
        glDeleteShader(shaders[i]);

}

void ShaderProgram::createFromFile(const std::string & vertex_shader_file,
                                   const std::string & fragment_shader_file,
                                   const std::string & geometry_shader_file,
                                   const std::string & tess_control_shader_file,
                                   const std::string & tess_evaluation_shader_file)
{
    std::string shader_files[5] = { vertex_shader_file, fragment_shader_file, geometry_shader_file, tess_control_shader_file, tess_evaluation_shader_file };
    std::string shader_strs[5];

    for(int i = 0; i < 5; ++i)
    {
        if (shader_files[i].empty() == true)
            continue;

        std::ifstream input_file(shader_files[i]);

        if(input_file.fail() == true)
        {
            std::cerr << "error: can't open file " << shader_files[i] << std::endl;
            std::exit(EXIT_FAILURE);
        }

        std::stringstream sstream;
        sstream << input_file.rdbuf();

        input_file.close();

        shader_strs[i] = sstream.str();

    }

    this->createFromCStyleString(shader_strs[0].empty() ? nullptr : shader_strs[0].c_str(),
                                 shader_strs[1].empty() ? nullptr : shader_strs[1].c_str(),
                                 shader_strs[2].empty() ? nullptr : shader_strs[2].c_str(),
                                 shader_strs[3].empty() ? nullptr : shader_strs[3].c_str(),
                                 shader_strs[4].empty() ? nullptr : shader_strs[4].c_str());

}


void ShaderProgram::createFromString(const std::string & vertex_shader_str,
                                     const std::string & fragment_shader_str,
                                     const std::string & geometry_shader_str,
                                     const std::string & tess_control_shader_str,
                                     const std::string & tess_evaluation_shader_str)
{

    this->createFromCStyleString(vertex_shader_str.empty() ?          nullptr : vertex_shader_str.c_str(),
                                 fragment_shader_str.empty() ?        nullptr : fragment_shader_str.c_str(),
                                 geometry_shader_str.empty() ?        nullptr : geometry_shader_str.c_str(),
                                 tess_control_shader_str.empty() ?    nullptr : tess_control_shader_str.c_str(),
                                 tess_evaluation_shader_str.empty() ? nullptr : tess_evaluation_shader_str.c_str());
}


void ShaderProgram::destory()
{
    glVerify(glDeleteProgram(this->program_));
    this->program_ = 0;
}

void ShaderProgram::use() const
{
    if (this->isValid())
    {
        glVerify(glUseProgram(this->program_));
    }
    else
        throw PhysikaException("error: invalid shader program!");
}

void ShaderProgram::unUse() const
{
    glVerify(glUseProgram(0));
}

void ShaderProgram::check() const
{
    if (this->isValid() == false)
        throw PhysikaException("error: invalid shader program!");

    GLint cur_program_id = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, &cur_program_id);

    if (cur_program_id != this->program_)
        throw PhysikaException("error: this shader_program is not binded, please call use() before setting");

}

bool ShaderProgram::setBool(const std::string & name, bool val) 
{
    return this->setInt(name, val);
}

bool ShaderProgram::setInt(const std::string & name, int val) 
{
    check();
    int location = glGetUniformLocation(this->program_, name.c_str());
    if (location == -1)
        return false;

    glVerify(glUniform1i(location, val));
    return true;
}

bool ShaderProgram::setFloat(const std::string & name, float val)
{
    check();
    int location = glGetUniformLocation(this->program_, name.c_str());
    if (location == -1)
        return false;

    glVerify(glUniform1f(location, val));
    return true;
}

bool ShaderProgram::setVec2(const std::string & name, const Vector2f & val)
{
    glm::vec2 glm_val = { val[0], val[1] };
    return this->setVec2(name, glm_val);
}

bool ShaderProgram::setVec2(const std::string & name, const glm::vec2 & val)
{
    check();
    int location = glGetUniformLocation(this->program_, name.c_str());
    if (location == -1)
        return false;

    glVerify(glUniform2fv(location, 1, glm::value_ptr(val)));
    return true;
}

bool ShaderProgram::setVec2(const std::string & name, float x, float y)
{
    check();
    int location = glGetUniformLocation(this->program_, name.c_str());
    if (location == -1)
        return false;

    glVerify(glUniform2f(location, x, y));
    return true;
}

bool ShaderProgram::setVec3(const std::string & name, const Vector3f & val)
{
    glm::vec3 glm_val = { val[0], val[1], val[2] };
    return this->setVec3(name, glm_val);
}

bool ShaderProgram::setVec3(const std::string & name, const glm::vec3 & val)
{
    check();
    int location = glGetUniformLocation(this->program_, name.c_str());
    if (location == -1)
        return false;

    glVerify(glUniform3fv(location, 1, glm::value_ptr(val)));
    return true;
}

bool ShaderProgram::setVec3(const std::string & name, float x, float y, float z)
{
    check();
    int location = glGetUniformLocation(this->program_, name.c_str());
    if (location == -1)
        return false;

    glVerify(glUniform3f(location, x, y, z));
    return true;
}

bool ShaderProgram::setVec4(const std::string & name, const Vector4f & val)
{
    glm::vec4 glm_val = { val[0], val[1], val[2], val[3] };
    return this->setVec4(name, glm_val);
}

bool ShaderProgram::setVec4(const std::string & name, const glm::vec4 & val)
{
    check();
    int location = glGetUniformLocation(this->program_, name.c_str());
    if (location == -1)
        return false;

    glVerify(glUniform4fv(location, 1, glm::value_ptr(val)));
    return true;
}

bool ShaderProgram::setVec4(const std::string & name, float x, float y, float z, float w)
{
    check();
    int location = glGetUniformLocation(this->program_, name.c_str());
    if (location == -1)
        return false;

    glVerify(glUniform4f(location, x, y, z, w));
    return true;
}

bool ShaderProgram::setMat2(const std::string & name, const Matrix2f & val)
{
    glm::mat2 glm_mat = {val(0,0), val(1,0),  //col 0
                         val(0,1), val(1,1)}; //col 1

    return this->setMat2(name, glm_mat);
}

bool ShaderProgram::setMat2(const std::string & name, const glm::mat2 & val)
{
    check();
    int location = glGetUniformLocation(this->program_, name.c_str());
    if (location == -1)
        return false;

    glVerify(glUniformMatrix2fv(location, 1, GL_FALSE, glm::value_ptr(val)));
    return true;
}

bool ShaderProgram::setMat3(const std::string & name, const Matrix3f & val)
{
    glm::mat3 glm_mat = { val(0,0), val(1,0), val(2,0),    //col 0
                          val(0,1), val(1,1), val(2,1),    //col 1
                          val(0,2), val(1,2), val(2,2)};   //col 2

    return this->setMat3(name, glm_mat);
}

bool ShaderProgram::setMat3(const std::string & name, const glm::mat3 & val)
{
    check();
    int location = glGetUniformLocation(this->program_, name.c_str());
    if (location == -1)
        return false;

    glVerify(glUniformMatrix3fv(location, 1, GL_FALSE, glm::value_ptr(val)));
    return true;
}

bool ShaderProgram::setMat4(const std::string & name, const Matrix4f & val)
{
    glm::mat4 glm_mat = { val(0,0), val(1,0), val(2,0), val(3,0),   //col 0
                          val(0,1), val(1,1), val(2,1), val(3,1),   //col 1
                          val(0,2), val(1,2), val(2,2), val(3,2),   //col 2
                          val(0,3), val(1,3), val(2,3), val(3,3)};  //col 3

    return this->setMat4(name, glm_mat);
}

bool ShaderProgram::setMat4(const std::string & name, const glm::mat4 & val)
{
    check();
    int location = glGetUniformLocation(this->program_, name.c_str());
    if (location == -1)
        return false;

    glVerify(glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(val)));
    return true;
}

bool ShaderProgram::isValid() const
{
    return glIsProgram(this->program_);
}

GLuint ShaderProgram::id() const
{
    return this->program_;
}

}// end of namespace Physika