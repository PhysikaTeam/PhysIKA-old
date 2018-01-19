/*
* @file shader.cpp
* @Brief Class Shader
* @author Wei Chen
*
* This file is part of Physika, a versatile physics simulation library.
* Copyright (C) 2013- Physika Group.
*
* This Source Code Form is subject to the terms of the GNU General Public License v2.0.
* If a copy of the GPL was not distributed with this file, you can obtain one at:
* http://www.gnu.org/licenses/gpl-2.0.html
*/

#include "shader.h"
#include <glm/glm.hpp>

#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"
#include "Physika_Render/Render_Scene_Config/render_scene_config.h"


namespace Physika{

void Shader::createFromCStyleString(const char * vertex_shader_source, 
                                    const char * fragment_shader_source, 
                                    const char * geometry_shader_source /* = nullptr */, 
                                    const char * tess_control_shader_source /* = nullptr */, 
                                    const char * tess_evaluation_shader_source /* = nullptr */)
{
    this->shader_prog_.createFromCStyleString(vertex_shader_source, 
                                              fragment_shader_source, 
                                              geometry_shader_source,
                                              tess_control_shader_source,
                                              tess_evaluation_shader_source);
}

void Shader::createFromFile(const std::string & vertex_shader_file, 
                            const std::string & fragment_shader_file, 
                            const std::string & geometry_shader_file /* =  */,
                            const std::string & tess_control_shader_file /* =  */, 
                            const std::string & tess_evaluation_shader_file /* = */ )
{
    this->shader_prog_.createFromFile(vertex_shader_file, 
                                      fragment_shader_file, 
                                      geometry_shader_file, 
                                      tess_control_shader_file, 
                                      tess_evaluation_shader_file);
}

void Shader::createFromString(const std::string & vertex_shader_str, 
                              const std::string & fragment_shader_str, 
                              const std::string & geometry_shader_str /* =  */, 
                              const std::string & tess_control_shader_str /* =  */, 
                              const std::string & tess_evaluation_shader_str /* = */ )
{
    this->createFromString(vertex_shader_str,
                           fragment_shader_str,
                           geometry_shader_str, 
                           tess_control_shader_str, 
                           tess_evaluation_shader_str);
}


void Shader::bindAndConfigBasicUniforms()
{
    this->bind();
    this->configCameraUniforms();
    this->configLightUniforms();
}

void Shader::bind()
{
    this->shader_prog_.use();
}

void Shader::configCameraUniforms()
{
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
    Camera<double> & camera = render_scene_config.camera();
    camera.configCameraToCurBindShader();
}

void Shader::configLightUniforms()
{
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
    LightManager & light_manager = render_scene_config.lightManager();
    light_manager.configLightsToCurBindShader();
}

void Shader::unBind() const
{
    this->shader_prog_.unUse();
}

}//end of namespace Physika