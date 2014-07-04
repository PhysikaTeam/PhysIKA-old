/*
 * @file light.cpp 
 * @Brief a light class for OpenGL.
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */
#include <iostream>
#include "light.h"
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"

namespace Physika
{

Light::Light()
{
    this->light_id_ = 0;
    std::cerr<<"warning: light_id are not specfied, light_id will be set to 0 !"<<std::endl;
}

Light::Light(GLenum light_id)
{
    this->light_id_ = light_id;
}

Light::~Light(){}

void Light::setLightId(GLenum light_id)
{
    this->light_id_ = light_id;
}

GLenum Light::lightId() const
{
    return this->light_id_;
}

void Light::turnOn()const
{
    glEnable(GL_LIGHTING);
    glEnable(this->light_id_);
}

void Light::turnOff()const
{
    glDisable(this->light_id_);
}

void Light::printInfo() const
{
    std::cout<<"light_id: "<<this->lightId()<<std::endl;
    std::cout<<"ambient: " <<this->ambient<float>()<<std::endl;
    std::cout<<"diffuse: " <<this->diffuse<float>()<<std::endl;
    std::cout<<"specular: "<<this->specular<float>()<<std::endl;
    std::cout<<"position: "<<this->position<float>()<<std::endl;
    std::cout<<"constantAttenation: " <<this->constantAttenuation<float>()<<std::endl;
    std::cout<<"linearAttenation: "<<this->linearAttenuation<float>()<<std::endl;
    std::cout<<": "<<this->quadraticAttenuation<float>()<<std::endl;
    std::cout<<"isLightOn: "<<this->isLightOn()<<std::endl;
}

bool Light::isLightOn() const
{
    unsigned char isOn;
    glGetBooleanv(this->light_id_, &isOn);
    if(isOn == GL_TRUE)
        return true;
    else
        return false;
}

std::ostream &  operator<< (std::ostream& out, const Light& light)
{
    out<<"light_id: "<<light.lightId()<<std::endl;
    out<<"ambient: " <<light.ambient<float>()<<std::endl;
    out<<"diffuse: " <<light.diffuse<float>()<<std::endl;
    out<<"specular: "<<light.specular<float>()<<std::endl;
    out<<"position: "<<light.position<float>()<<std::endl;
    out<<"constantAttenation: " <<light.constantAttenuation<float>()<<std::endl;
    out<<"linearAttenation: "<<light.linearAttenuation<float>()<<std::endl;
    out<<"quadraticAttenuation: "<<light.quadraticAttenuation<float>()<<std::endl;
    out<<"isLightOn: "<<light.isLightOn()<<std::endl;
    return out;
}

}// end of namespace Physika
