/*
 * @file light_base.h 
 * @brief class LightBase.
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

#include <string>

#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Render/Color/color.h"

namespace Physika{

enum class LightType
{
    DIRECTIONAL_LIGHT,
    POINT_LIGHT,
    SPOT_LIGHT,
    FLEX_SPOT_LIGHT
};

class LightBase
{
public:
    LightBase() = default;
    LightBase(const Color4f & ambient_col, const Color4f & diffuse_col, const Color4f & specular_col);

    LightBase(const LightBase &) = default;
    LightBase & operator = (const LightBase &) = default;

    virtual ~LightBase() = default;

    virtual LightType type() const = 0;
    virtual void configToCurBindShader(const std::string & light_str) = 0;

    //getter
    const Color4f &  ambient() const;
    const Color4f &  diffuse() const;
    const Color4f &  specular() const;

    //setter
    void setAmbient(const Color4f & ambient);
    void setDiffuse(const Color4f & diffuse);
    void setSpecular(const Color4f & specular);

    void enableLighting();
    void disableLighting();
    bool isEnableLighting() const;

private:
    Color4f ambient_col_ = {0.0f, 0.0f, 0.0f};
    Color4f diffuse_col_ = {1.0f, 1.0f, 1.0f};
    Color4f specular_col_ = {1.0f, 1.0f, 1.0f};

    bool enable_lighting_ = true; //hook to enable/disable lighting
};
    
}//end of namespace Physika