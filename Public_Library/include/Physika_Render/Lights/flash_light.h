/*
 * @file flash_light.h 
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

#pragma  once

#include "spot_light.h"
#include "Physika_Render/Render_Scene_Config/render_scene_config.h"

namespace Physika{

/*
 *Note: A flashlight is a spotlight located at the viewer's position and usually aimed straight ahead from the player's perspective.
 *      Basically a flashlight is a normal spotlight, but its position and spot direction continually updated based on player's position and orientation.
 */

class FlashLight: public SpotLight
{
public:

    const Vector3f pos() const override;
    const Vector3f spotDirection() const override;

private:
    //disable setPos & setSpotDirection
    using SpotLight::setPos;
    using SpotLight::setSpotDirection;

private:
    RenderSceneConfig * render_scene_config_ = &RenderSceneConfig::getSingleton();
};

}//end of namespace Physika