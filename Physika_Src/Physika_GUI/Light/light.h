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

namespace Physika
{

template<typename Scalar>
class Light
{
public:
    enum LightType
    {
        POSITION_BASE,
        DIRECTION_BASE;
    };
public:
    // construction and destruction
    Light();
    ~Light();

    // public interface: setter and getter
    void setToPositionBase();
    void setToDirectionBase();

    void             setAmbient(const Color & color);
    Color<Scalar>    ambient();
    void             setDiffuse(const Color & color);
    Color<Scalar>    Diffuse();
    void             setSpecular(const Color & color);
    Color<Scalar>    Specular();

    void             setPosOrDir(const Vector<Scalar,3>& pos_or_dir);
    Vector<Scalar,3> PosOrDir();

    void             setSpotDirection(const Vector<Scalar,3>& direction);
    Vector<Scalar,3> spotDirection();

    void             setSpotExponent(Scalar exponent);
    Scalar           spotExpoent();

    void             setSpotCutoff(Sclar cutoff);
    Scalar           spotCutoff();

    void             setConstantAttenation(Scalar constant_atten)
    Scalar           ConstantAttenation();
    void             setLinearAttenation(Scalar linear_atten);
    Scalar           LinearAttenation();
    void             setQuadraticAttenation(Scalar quad_atten);
    Scalar           QuadraticAttenation();

    void             setLightModelAmbient(const Color<Scalar> color);
    Color<Scalar>    LightModelAmbient();
    void             setLigntModelLocalViewer(bool viewer);
    bool             LightModelLocalViewer();
    void             setLightModelTwoSide(bool two_size);
    bool             LightModelTwoSize();

    void             setLightModelColorControl(GLenum penum);
    GLenum           LightModelColorControl();

};

}      //end of namespace Physika
#endif //PHYSIKA_GUI_LIGHT_LIGHT_H_