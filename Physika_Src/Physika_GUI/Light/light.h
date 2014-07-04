/*
 * @file light.h 
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
#ifndef PHYSIKA_GUI_LIGHT_LIGHT_H_
#define PHYSIKA_GUI_LIGHT_LIGHT_H_

#include <GL/gl.h>
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Render/Color/color.h"
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"

namespace Physika{
//Note: Light is designed for float and integer type, since openGL functions about light was imlementated for these two.
//      Generally, type float is sufficient, and is strongly recommended.
//      In Light class, as you can see in head file, we store NO data member, EXCEPT the light_id (corresponding to GL_LIGHT0,
//      GL_LIGHT1, ......)to specify which light you are using from openGL.
//      Thus, every setter and getter is operated by directly call openGL set-function and get-function(such as glLight and 
//      glGetFloatv).
//      As a consequence, The Light class maintains exactly the feature of "state machine" of openGL, i.e. You can change 
//      the light object by call openGL functions directly outside, only if you have the right light_id.

class Light
{
public:
    // construction and destruction
    Light();
    explicit Light(GLenum light_id);
    virtual ~Light();

    // public interface: setter and getter
    void    setLightId(GLenum light_id);
    GLenum  lightId() const;

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
    void             setPosition(const Vector<Scalar,3>& pos);
    template <typename Scalar>
    Vector<Scalar,3> position() const;
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
protected:
    GLenum light_id_;
};

std::ostream &  operator<< (std::ostream& out, const Light& light);

}      //end of namespace Physika

#include "light-inl.h"
#endif //PHYSIKA_GUI_LIGHT_LIGHT_H_