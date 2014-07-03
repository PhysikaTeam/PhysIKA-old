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
    // construction and destruction
    Light();
    virtual ~Light();

    // public interface: setter and getter
 
    virtual void             setAmbient(const Color<Scalar> & color);
    virtual Color<Scalar>    ambient();
    virtual void             setDiffuse(const Color<Scalar> & color);
    virtual Color<Scalar>    diffuse();
    virtual void             setSpecular(const Color<Scalar> & color);
    virtual Color<Scalar>    Specular();

    virtual void             setPosition(const Vector<Scalar,3>& pos_or_dir);
    virtual Vector<Scalar,3> position();

    virtual void             setSpotDirection(const Vector<Scalar,3>& direction);
    virtual Vector<Scalar,3> spotDirection();

    virtual void             setSpotExponent(Scalar exponent);
    virtual Scalar           spotExpoent();

    virtual void             setSpotCutoff(Sclar cutoff);
    virtual Scalar           spotCutoff();

    virtual void             setConstantAttenation(Scalar constant_atten)
    virtual Scalar           constantAttenation();
    virtual void             setLinearAttenation(Scalar linear_atten);
    virtual Scalar           linearAttenation();
    virtual void             setQuadraticAttenation(Scalar quad_atten);
    virtual Scalar           quadraticAttenation();

    virtual void             setLightModelAmbient(const Color<Scalar> color);
    virtual Color<Scalar>    lightModelAmbient();
    virtual  void             setLigntModelLocalViewer(bool viewer);
    virtual bool             LightModelLocalViewer();
    virtual void             setLightModelTwoSide(bool two_size);
    virtual bool             lightModelTwoSize();

    virtual void             setLightModelColorControl(GLenum penum);
    virtual GLenum           lightModelColorControl();
protected:
    GLenum light_id_;
};

}      //end of namespace Physika
#endif //PHYSIKA_GUI_LIGHT_LIGHT_H_