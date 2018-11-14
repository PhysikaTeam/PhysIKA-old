/*
 * @file light_manager.cpp 
 * @Brief maintains a list of lights.
 * @author Wei Chen, Fei Zhu
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
#include <memory>

#include "Physika_Core/Utilities/physika_assert.h"

#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"
#include "Physika_Render/Lights/light_manager.h"
#include "Physika_Render/Lights/light_base.h"
#include "Physika_Render/Lights/flash_light.h"

using std::list;

namespace Physika{

void LightManager::configLightsToCurBindShader()
{
    int directional_light_num = 0;
    int point_light_num = 0;
    int spot_light_num = 0;
    int flex_spot_light_num = 0;

    for (int i = 0; i < this->numLights(); ++i)
    {
        const std::shared_ptr<LightBase> & light = this->lightAtIndex(i);

        //if light is disabled, then skip it
        if (light->isEnableLighting() == false)
            continue;

        if (light->type() == LightType::DIRECTIONAL_LIGHT)
        {
            std::string light_str = "directional_lights[" + std::to_string(directional_light_num) + "]";
            light->configToCurBindShader(light_str);

            ++directional_light_num;
        }
        else if (light->type() == LightType::POINT_LIGHT)
        {
            std::string light_str = "point_lights[" + std::to_string(point_light_num) + "]";
            light->configToCurBindShader(light_str);

            ++point_light_num;
        }
        else if (light->type() == LightType::SPOT_LIGHT)
        {
            std::string light_str = "spot_lights[" + std::to_string(spot_light_num) + "]";
            light->configToCurBindShader(light_str);

            ++spot_light_num;
        }
        else if (light->type() == LightType::FLEX_SPOT_LIGHT)
        {
            std::string light_str = "flex_spot_lights[" + std::to_string(flex_spot_light_num) + "]";
            light->configToCurBindShader(light_str);

            ++flex_spot_light_num;
        }
        else
        {
            throw PhysikaException("error: unknown Light Type");
        }
    }

    openGLSetCurBindShaderInt("directional_light_num", directional_light_num);
    openGLSetCurBindShaderInt("point_light_num", point_light_num);
    openGLSetCurBindShaderInt("spot_light_num", spot_light_num);
    openGLSetCurBindShaderInt("flex_spot_light_num", flex_spot_light_num);
}

unsigned int LightManager::numLights()const
{
    return this->light_list_.size();
}

void LightManager::insertBack(std::shared_ptr<LightBase> light_p)
{
    if(light_p == nullptr)
    {
        std::cerr<<"error: Cannot insert NULL light to LightManager, operation will be ignored!"<<std::endl;
        return ;
    }

    if (this->lightIndex(light_p) != -1)
    {
        std::cerr << "error: this light is already in LightManager, its index is " << this->lightIndex(light_p) << ", operation will be ignored!" << std::endl;
        return;
    }

    this->light_list_.push_back(std::move(light_p));
}

void LightManager::insertFront(std::shared_ptr<LightBase> light_p)
{
    if(light_p == nullptr)
    {
        std::cerr<<"error: Cannot insert NULL light to LightManager, operation will be ignored!"<<std::endl;
        return ;
    }
    
    if (this->lightIndex(light_p) != -1)
    {
        std::cerr << "error: this light is already in LightManager, its index is " << this->lightIndex(light_p) << ", operation will be ignored!" << std::endl;
        return;
    }

    this->light_list_.push_front(std::move(light_p));
}

void LightManager::insertAtIndex(unsigned int index, std::shared_ptr<LightBase> light_p)
{
    bool index_valid = (index<light_list_.size());
    if(!index_valid)
    {
        std::cerr<<"Light index out of range, operation will be ignored!\n";
        return ;
    }
    
    if(light_p)
    {
        if(this->lightIndex(light_p) != -1)
        {
            std::cerr<<"error: this light is already in LightManager, its index is "<<this->lightIndex(light_p)<<", operation will be ignored!"<<std::endl;
            return ;
        }

        auto pos = light_list_.begin();
        while (index != 0)
        {
            --index;
            ++pos;
        }
        light_list_.insert(pos, light_p);

    }
    else
    {
        std::cerr << "error: cannot insert NULL light to LightManager, operation will be ignored!";
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
        std::cerr<<"error: light index out of range, operation will be ignored!!\n";
        return ;
    }
    auto pos = light_list_.begin();
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

std::shared_ptr<const LightBase> LightManager::lightAtIndex(unsigned int index) const
{
    bool index_valid = (index<light_list_.size());
    if(!index_valid)
    {
        std::cerr<<"error: light index out of range, NULL is returned! \n";
        return {};
    }
    auto iter = light_list_.begin();
    while(index != 0)
    {
        --index;
        ++iter;
    }
    return *iter;
}

std::shared_ptr<LightBase> LightManager::lightAtIndex(unsigned int index)
{
    bool index_valid = (index<light_list_.size());
    if(!index_valid)
    {
        std::cerr<<"error: Light index out of range, NULL is returned!!\n";
        return {};
    }
    
    auto iter = light_list_.begin();
    while(index != 0)
    {
        --index;
        ++iter;
    }

    return *iter;
}

int LightManager::lightIndex(const std::shared_ptr< const LightBase> & light_p) const
{
    if(light_p == nullptr)
        return -1;
    auto iter = light_list_.begin();
    int index = 0;
    while(iter != light_list_.end())
    {
        if(*iter == light_p)
            return index;
        ++iter;
        ++index;
    }
    return -1;
}

void LightManager::turnAllOn()
{
    auto iter = light_list_.begin();
    while(iter != light_list_.end())
    {
        PHYSIKA_ASSERT(*iter);
        (*iter)->enableLighting();
        ++iter;
    }
}

void LightManager::turnAllOff()
{
    auto iter = light_list_.begin();
    while(iter != light_list_.end())
    {
        PHYSIKA_ASSERT(*iter);
        (*iter)->disableLighting();
        iter++;
    }
}

void LightManager::turnLightOnAtIndex(unsigned int index)
{
    bool index_valid = (index<light_list_.size());
    if(!index_valid)
    {
        std::cerr<<"error: light index out of range, operation will be ignored!!\n";
        return ;
    }

    auto iter = light_list_.begin();
    while(index != 0)
    {
        iter++;
        index--;
    }
    PHYSIKA_ASSERT(*iter);
    (*iter)->enableLighting();
}

void LightManager::turnLightOffAtIndex(unsigned int index)
{
    bool index_valid = (index<light_list_.size());
    if(!index_valid)
    {
        std::cerr<<"error: light index out of range, operation will be ignored!!\n";
        return ;
    }
    
    auto iter = light_list_.begin();
    while(index != 0)
    {
        iter++;
        index--;
    }
    PHYSIKA_ASSERT(*iter);
    (*iter)->disableLighting();
}


void LightManager::printInfo()
{
    std::cout<<"light number: "<<this->numLights()<<std::endl;
    for(unsigned int i=0; i<this->numLights();i++)
    {
        std::cout<<"light id: "<< i <<"  state: "<< this->lightAtIndex(i)->isEnableLighting()<<std::endl;
    }
}

std::ostream& operator << (std::ostream& out, const LightManager & light_manager)
{
    out<<"light number: "<<light_manager.numLights()<<std::endl;
    for(unsigned int i=0; i<light_manager.numLights();i++)
    {
        out<<"light id: "<<i<<"  state: "<< (light_manager.lightAtIndex(i))->isEnableLighting() <<std::endl;
    }
    return out;
}

}//end of namespace Physika


