/*
 * @file directional_light.h 
 * @brief class DirectionalLight.
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

#include "light_base.h"

namespace Physika{

class DirectionalLight: public LightBase
{
public:
    DirectionalLight() = default;

    DirectionalLight(const Color4f & ambient_col, 
                     const Color4f & diffuse_col, 
                     const Color4f & specular_col,
                     const Vector3f & direction);

    LightType type() const override;
    void configToCurBindShader(const std::string & light_str) override;

    const Vector3f & direction() const;
    void setDirection(const Vector3f & direction);

private:
    Vector3f direction_ = {0.0f,  0.0f, 1.0f};
};
    
}//end of namespace Physika