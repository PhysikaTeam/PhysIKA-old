/*
 * @file render_manager.h 
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

#ifndef PHYSIKA_RENDER_LIGHTS_LIGHT_MANAGER_H_
#define PHYSIKA_RENDER_LIGHTS_LIGHT_MANAGER_H_

#include <list>
#include <GL/gl.h>
#include "Physika_Render/Color/color.h"
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"

/*
 * Note1: Since OpegnGL demands that each graphic card has to implement at least 8 lights, to avoid undefined situation where lights number are greater than 8,
 *        LightManager only perserve 8 Light Object(actually pointer) in list at most.
 * Note2: LightManager is also responsible for configuring LIGHT MODEL, since Light Model actually belongs to entire scenario but not only 
 *        one particular Light. The configuring consists of two parameters:
 *        1.GL_LIGHT_MODEL_LOCAL_VIEWER
 *        2.GL_LIGHT_MODEL_TWO_SIDE 
 *        3.GL_LIGHT_MODEL_COLOR_CONTROL: but this parameter is not supported in windows, so we ingnore it.
 *        We have corresponing setter and getter for these parameters.
 */

namespace Physika{

class Light;
class LightManager
{
public:
    LightManager();
    ~LightManager();
    unsigned int numLights() const;  //number of lights in list;
    void insertBack(Light*);         //insert new light at back of list
    void insertFront(Light*);        //insert new light at front of list
    void insertAtIndex(unsigned int index, Light *light);      //insert new light before the index-th light
    void removeBack();                                        //remove light at back of light list
    void removeFront();                                       //remove light at front of light list
    void removeAll();
    void removeAtIndex(unsigned int index);                   //remove the index-th light in list

    const Light* lightAtIndex(unsigned int index) const;       //return pointer to the light at given index
    Light* lightAtIndex(unsigned int index);
    int lightIndex(Light *light) const;                        //return index of light in list, if light not in queue, return -1

    void turnAllOn();                                         //turn all lights on
    void turnAllOff();                                        //turn all lights off
    void turnLightOnAtIndex(unsigned int index);              // turn light at given index in list ON
    void turnLightOffAtIndex(unsigned int index);             // turn light at given index in list Off
    
    //Put all the lights that are turned on into use, called after the model view matrix are setup
    void lightScene();                                        

    template <typename ColorType>
    void             setLightModelAmbient(const Color<ColorType> &color);
    template <typename ColorType>
    Color<ColorType> lightModelAmbient() const;

    void    setLightModelLocalViewer(bool viewer);
    bool    lightModelLocalViewer() const;
    void    setLightModelTwoSide(bool two_size);
    bool    lightModelTwoSize() const;
    //void    setLightModelColorControl(GLenum penum);
    //GLenum  lightModelColorControl() const;

    void    printInfo();

protected:
    std::list<Light*> light_list_;
};

template<typename ColorType>
void LightManager::setLightModelAmbient(const Color<ColorType> &color)
{
    openGLLightModelAMBient(color);
}

template<typename ColorType>
Color<ColorType> LightManager::lightModelAmbient() const
{
    float color[4];
    glGetFloatv(GL_LIGHT_MODEL_AMBIENT,color);
    Color<float> temp_color(color[0], color[1], color[2], color[3]);
    return temp_color.convertColor<ColorType>();
}

//declaration of << operator
std::ostream& operator << (std::ostream& out, const LightManager & light_manager);

}  //end of namespace Physika

#endif //PHYSIKA_RENDER_LIGHTS_LIGHT_MANAGER_H_
