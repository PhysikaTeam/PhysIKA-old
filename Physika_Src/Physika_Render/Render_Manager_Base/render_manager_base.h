/*
* @file render_manager_base.h
* @Basic class RenderManagerBase
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

#include "Physika_Render/Render_Scene_Config/render_scene_config.h"

namespace Physika {

class RenderManagerBase
{
public:
    virtual ~RenderManagerBase() = default;

    virtual void render() = 0;

protected:
    RenderSceneConfig * render_scene_config_ = &RenderSceneConfig::getSingleton();
};

}//end of namespace Physika