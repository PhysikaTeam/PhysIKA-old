/*
 * @file render_manager.h 
 * @Brief maintains a list of lights.
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

#include <list>
#include <memory>

namespace Physika{

class LightBase;

class LightManager
{
public:
    LightManager() = default;
    ~LightManager() = default;

    void configLightsToCurBindShader();

    unsigned int numLights() const;                                                 //number of lights in list;
    void insertBack(std::shared_ptr<LightBase>);                                    //insert new light at back of list
    void insertFront(std::shared_ptr<LightBase>);                                   //insert new light at front of list
    void insertAtIndex(unsigned int index, std::shared_ptr<LightBase>);             //insert new light before the index-th light
    void removeBack();                                                              //remove light at back of light list
    void removeFront();                                                             //remove light at front of light list
    void removeAll();                                                               //remove All lights
    void removeAtIndex(unsigned int index);                                         //remove the index-th light in list

    std::shared_ptr<const LightBase> lightAtIndex(unsigned int index) const;       //return pointer to the light at given index
    std::shared_ptr<LightBase> lightAtIndex(unsigned int index);                   //return pointer to the light at given index
    int lightIndex(const std::shared_ptr< const LightBase> &) const;               //return index of light in list, if light not in queue, return -1

    void turnAllOn();                                          //turn all lights on
    void turnAllOff();                                         //turn all lights off
    void turnLightOnAtIndex(unsigned int index);               //turn light at given index in list ON
    void turnLightOffAtIndex(unsigned int index);              //turn light at given index in list Off
                                          
    void  printInfo();

protected:
    std::list<std::shared_ptr<LightBase>> light_list_;
};


//declaration of << operator
std::ostream& operator << (std::ostream& out, const LightManager & light_manager);

}  //end of namespace Physika


