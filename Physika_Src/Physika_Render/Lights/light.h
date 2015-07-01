/*
 * @file light.h 
 * @brief a light class for OpenGL.
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

#ifndef PHYSIKA_RENDER_LIGHTS_LIGHT_H_
#define PHYSIKA_RENDER_LIGHTS_LIGHT_H_

#include <GL/gl.h>
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Render/Color/color.h"
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"

namespace Physika{

/*
 * Light class only stores its id and position, other properties of lights
 * are directly updated to OpenGL via corresponding methods
 * Note: Light is turned on doesn't mean it's put into use, lightScene() must be
 *       called after the model view matrix set up in OpenGL code
 * 
 * Usage: 
 *       1. Define Light object
 *       2. Set its properties
 *       3. Call lightScene() in your OpenGL code, after you'v set up the model view matrix
 *          of your camera
 *
 * 8 lights are supported at most.
 */

class Light
{
public:
    // construction and destruction
    Light();
    explicit Light(GLenum light_id);
    Light(GLenum light_id, const Vector<float,3> &light_position);
    ~Light();

    //call this method in your OpenGL code after the model view matrix is set up
    void lightScene();

    // public interface: setter and getter
    void    setLightId(GLenum light_id);
    GLenum  lightId() const;
    void    setPosition(const Vector<float,3> &pos);
    const Vector<float,3>& position() const;
    Vector<float,3>& position();

    template <typename ColorType>
    void             setAmbient(const Color<ColorType> & color);
    template <typename ColorType>
    Color<ColorType> ambient() const;
    template <typename ColorType>
    void             setDiffuse(const Color<ColorType> & color);
    template <typename ColorType>
    Color<ColorType> diffuse() const;
    template <typename ColorType>
    void             setSpecular(const Color<ColorType> & color);
    template <typename ColorType>
    Color<ColorType> specular() const;

    template <typename Scalar>
    void             setConstantAttenuation(Scalar constant_atten);
    template <typename Scalar>
    Scalar           constantAttenuation() const;
    template <typename Scalar>
    void             setLinearAttenuation(Scalar linear_atten);
    template <typename Scalar>
    Scalar           linearAttenuation() const;
    template <typename Scalar>
    void             setQuadraticAttenuation(Scalar quad_atten);
    template <typename Scalar>
    Scalar           quadraticAttenuation() const;

    //Note: turnOn will Call glEnable(GL_LIGHTING), while turnOff will NOT call glDisable(GL_LIGHTING).
    void turnOn() const;
    void turnOff() const;
    bool isLightOn() const;
    void printInfo() const;
	static void printOccupyInfo();
protected:
    void createOneLight();  //create one light with a random avaible id
    //disable copy
    Light(const Light &light);
    Light& operator= (const Light &light);
protected:
    GLenum light_id_;
    Vector<float,3> light_position_;
	static bool is_occupied_[8];
};

std::ostream &  operator<< (std::ostream& out, const Light& light);

}      //end of namespace Physika

#include "Physika_Render/Lights/light-inl.h"

#endif //PHYSIKA_RENDER_LIGHTS_LIGHT_H_
