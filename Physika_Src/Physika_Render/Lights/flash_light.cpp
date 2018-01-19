/*
 * @file flash_light.cpp
 * @Brief class FlashLight
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

#include <iostream>
#include "flash_light.h"
#include "Physika_Core/Utilities/physika_exception.h"

namespace Physika{

const Vector3f FlashLight::pos() const
{
    const Vector3d pos = render_scene_config_->cameraPosition();
    return Vector3f(pos[0], pos[1], pos[2]);
}

const Vector3f FlashLight::spotDirection() const
{
    Vector3d spot_direction = render_scene_config_->cameraFocusPosition() - render_scene_config_->cameraPosition();
    spot_direction.normalize();
    return Vector3f(spot_direction[0], spot_direction[1], spot_direction[2]);
}

    
}//end of namespace Physika