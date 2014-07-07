/*
 * @file spot_light.cpp 
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
#include "Physika_GUI/Lights/spot_light.h"

namespace Physika
{

SpotLight::SpotLight():Light(){}
SpotLight::SpotLight(GLenum light_id):Light(light_id){}

SpotLight::~SpotLight(){}

}// end of namespace Physika
