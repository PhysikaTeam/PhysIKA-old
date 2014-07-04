/*
 * @file spot_light.h 
 * @Brief a spot light class for OpenGL.
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
#ifndef PHYSIKA_GUI_LIGHTS_SPOT_LIGHT_H_
#define PHYSIKA_GUI_LIGHTS_SPOT_LIGHT_H_

#include "Physika_GUI/Lights/light.h"

namespace Physika{

class SpotLight: public Light
{
public:
    // construction and destruction
    SpotLight();
    explicit SpotLight(GLenum light_id);
    virtual ~SpotLight();
    template<typename Scalar> void setSpotDirection(const Vector<Scalar,3>& direction);
    template<typename Scalar> Vector<Scalar,3> spotDirection() const;
    template<typename Scalar> void setSpotExponent(Scalar exponent);
    template<typename Scalar> Scalar spotExponent() const;
    template<typename Scalar> void setSpotCutoff(Scalar cutoff);
    template<typename Scalar> Scalar spotCutoff() const;
};

} //end of namespace Physika

#include "Physika_GUI/Lights/spot_light-inl.h"

#endif //PHYSIKA_GUI_LIGHTS_SPOT_LIGHT_H_
