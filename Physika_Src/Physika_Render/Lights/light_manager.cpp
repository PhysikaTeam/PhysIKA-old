/*
 * @file light_manager.cpp 
 * @Brief maintains a list of lights.
 * @author Wei Chen, Fei Zhu
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
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Render/Lights/light_manager.h"
#include "Physika_Render/Lights/light.h"

using std::list;

namespace Physika
{

LightManager::LightManager(){}

LightManager::~LightManager(){}

unsigned int LightManager::numLights()const
{
    return this->light_list_.size();
}

void LightManager::insertBack(Light * light_p)
{
    if(light_p == NULL)
    {
        std::cerr<<"error: Cannot insert NULL light to LightManager, operation will be ignored!"<<std::endl;
        return ;
    }
	if(this->lightIndex(light_p) != -1)
	{
		std::cerr<<"error: this light is already in LightManager, its index is "<<this->lightIndex(light_p)<<", operation will be ignored!"<<std::endl;
        return ;
	}
    if(this->numLights()<8)
        this->light_list_.push_back(light_p);
    else
    {
        std::cerr<<"error: the length of light list will be greater than '8', we only preserve 8 light id at most, operation will be ignored!"<<std::endl;
        return ;
    }
}

void LightManager::insertFront(Light * light_p)
{
    if(light_p == NULL)
    {
        std::cerr<<"error: Cannot insert NULL light to LightManager, operation will be ignored!"<<std::endl;
        return ;
    }
	if(this->lightIndex(light_p) != -1)
	{
		std::cerr<<"error: this light is already in LightManager, its index is "<<this->lightIndex(light_p)<<", operation will be ignored!"<<std::endl;
        return ;
	}
    if(this->numLights()<8)
        this->light_list_.push_front(light_p);
    else
    {
        std::cerr<<"error: the length of light list will be greater than '8', we only preserve 8 light id at most, operation will be ignored! "<<std::endl;
        return ;
    }
}

void LightManager::insertAtIndex(unsigned int index, Light *light)
{
    bool index_valid = (index<light_list_.size());
    if(!index_valid)
    {
        std::cerr<<"Light index out of range, operation will be ignored!\n";
        return ;
    }
	
    if(light)
    {
		if(this->lightIndex(light) != -1)
		{
			std::cerr<<"error: this light is already in LightManager, its index is "<<this->lightIndex(light)<<", operation will be ignored!"<<std::endl;
			return ;
		}
        if(this->numLights()<8)
        {
            list<Light*>::iterator pos = light_list_.begin();
            while(index != 0)
            {
                --index;
                ++pos;
            }
            light_list_.insert(pos,light);
        }
        else
        {
            std::cerr<<"error: the length of light list will be greater than '8', we only perserve 8 light id at most, operation will be ignored! "<<std::endl;
            return ;
        }
    }
    else
    {
        std::cerr<<"Cannot insert NULL light to LightManager, operation ignored!\n";
    }
}

void LightManager::removeBack()
{
    light_list_.pop_back();
}

void LightManager::removeFront()
{
    light_list_.pop_front();
}

void LightManager::removeAtIndex(unsigned int index)
{
    bool index_valid = (index<light_list_.size());
    if(!index_valid)
    {
        std::cerr<<"light index out of range, operation will be ignored!!\n";
        return ;
    }
    list<Light*>::iterator pos = light_list_.begin();
    while(index != 0)
    {
        --index;
        ++pos;
    }
    light_list_.erase(pos);
}

void LightManager::removeAll()
{
    light_list_.clear();
}

const Light* LightManager::lightAtIndex(unsigned int index) const
{
    bool index_valid = (index<light_list_.size());
    if(!index_valid)
    {
        std::cerr<<"error: light index out of range, NULL is returned! \n";
        return NULL;
    }
    list<Light*>::const_iterator iter = light_list_.begin();
    while(index != 0)
    {
        --index;
        ++iter;
    }
    Light *cur_render = *iter;
    return cur_render;
}

Light* LightManager::lightAtIndex(unsigned int index)
{
    bool index_valid = (index<light_list_.size());
    if(!index_valid)
    {
        std::cerr<<"error: Light index out of range, NULL is returned!!\n";
        return NULL;
    }
    list<Light*>::iterator iter = light_list_.begin();
    while(index != 0)
    {
        --index;
        ++iter;
    }
    Light *cur_render = *iter;
    return cur_render;
}

int LightManager::lightIndex(Light *light) const
{
    if(light==NULL)
        return -1;
    list<Light*>::const_iterator iter = light_list_.begin();
    int index = 0;
    while(iter != light_list_.end())
    {
        if(*iter == light)
            return index;
        ++iter;
        ++index;
    }
    return -1;
}

void LightManager::turnAllOn()
{
    list<Light *>::iterator iter = light_list_.begin();
    while(iter != light_list_.end())
    {
        PHYSIKA_ASSERT(*iter);
        (*iter)->turnOn();
        iter++;
    }
}

void LightManager::turnAllOff()
{
    list<Light *>::iterator iter = light_list_.begin();
    while(iter != light_list_.end())
    {
        PHYSIKA_ASSERT(*iter);
        (*iter)->turnOff();
        iter++;
    }
}

void LightManager::turnLightOnAtIndex(unsigned int index)
{
    bool index_valid = (index<light_list_.size());
    if(!index_valid)
    {
        std::cerr<<"Light index out of range, operation will be ignored!!\n";
        return ;
    }
    list<Light *>::iterator iter = light_list_.begin();
    while(index != 0)
    {
        iter++;
        index--;
    }
    PHYSIKA_ASSERT(*iter);
    (*iter)->turnOn();
}

void LightManager::turnLightOffAtIndex(unsigned int index)
{
    bool index_valid = (index<light_list_.size());
    if(!index_valid)
    {
        std::cerr<<"Light index out of range, operation will be ignored!!\n";
        return ;
    }
    list<Light *>::iterator iter = light_list_.begin();
    while(index != 0)
    {
        iter++;
        index--;
    }
    PHYSIKA_ASSERT(*iter);
    (*iter)->turnOff();
}

void LightManager::lightScene()
{
    list<Light *>::iterator iter = light_list_.begin();
    while(iter != light_list_.end())
    {
        PHYSIKA_ASSERT(*iter);
        (*iter)->lightScene();
        iter++;
    }
}

void LightManager::setLightModelLocalViewer(bool viewer)
{
    openGLLightModel(GL_LIGHT_MODEL_LOCAL_VIEWER, viewer);
}

bool LightManager::lightModelLocalViewer()const
{
    unsigned char viewer;
    glGetBooleanv(GL_LIGHT_MODEL_LOCAL_VIEWER, &viewer);
    if(viewer == GL_FALSE)
        return false;
    else
        return true;
}

void LightManager::setLightModelTwoSide(bool two_side)
{
    openGLLightModel(GL_LIGHT_MODEL_TWO_SIDE, two_side);
}

bool LightManager::lightModelTwoSize()const
{
    unsigned char two_side;
    glGetBooleanv(GL_LIGHT_MODEL_TWO_SIDE, &two_side);
    if(two_side == GL_TRUE)
        return true;
    else
        return false;
}
/*
void LightManager::setLightModelColorControl(GLenum penum)
{
    openGLLightModel(GL_LIGHT_MODEL_COLOR_CONTRUL, penum);
}

GLenum LightManager::lightModelColorControl()const
{
    int color_control;
    glGetIntegerv(GL_LIGHT_COLOR_CONTROL, &color_control);
    return (GLenum)color_control;
}
*/

void LightManager::printInfo()
{
    std::cout<<"light number: "<<this->numLights()<<std::endl;
    for(unsigned int i=0; i<this->numLights();i++)
    {
        std::cout<<"light id: "<<this->lightAtIndex(i)->lightId()<<"  state: "<< this->lightAtIndex(i)->isLightOn()<<std::endl;
    }
    std::cout<<"light model ambient: "<<this->lightModelAmbient<float>()<<std::endl;
    std::cout<<"light model local viewer: "<<this->lightModelLocalViewer()<<std::endl;
    std::cout<<"light model two side: "<<this->lightModelTwoSize()<<std::endl;
    //std::cout<<"light model color control: "<<this->lightModelColorControl()<<std::endl;
}

std::ostream& operator << (std::ostream& out, const LightManager & light_manager)
{
    out<<"light number: "<<light_manager.numLights()<<std::endl;
    for(unsigned int i=0; i<light_manager.numLights();i++)
    {
        out<<"light id: "<<(light_manager.lightAtIndex(i))->lightId()<<"  state: "<< (light_manager.lightAtIndex(i))->isLightOn()<<std::endl;
    }
    out<<"light model ambient: "<<light_manager.lightModelAmbient<float>()<<std::endl;
    out<<"light model local viewer: "<<light_manager.lightModelLocalViewer()<<std::endl;
    out<<"light model two side: "<<light_manager.lightModelTwoSize()<<std::endl;
    return out;
}

}//end of namespace Physika


