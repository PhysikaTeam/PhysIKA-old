/*
 * @file flex_spot_light.h 
 * @brief class FlexSpotLight.
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

#include <glm/glm.hpp>

#include "Physika_Core/Vectors/vector_3d.h"
#include "light_base.h"

namespace Physika
{

/*Note: FlexSpotLight is a light type extracted from Flex demo codes, so we name it with prefix Flex.
 *      The design for this type of light still needs further consideration.
 */

class FlexSpotLight: public LightBase
{
public:
    FlexSpotLight() = default;
    FlexSpotLight(const Color4f & ambient_col, 
                  const Color4f & diffuse_col, 
                  const Color4f & specular_col,
                  Vector3f light_pos,
                  Vector3f light_target,
                  float light_fov,
                  float light_spot_min,
                  float light_spot_max);

    LightType type() const override;
    void configToCurBindShader(const std::string & light_str) override;

    glm::mat4 lightProjMatrix() const;
    glm::mat4 lightViewMatrix() const;
    glm::mat4 lightTransformMatrix() const;

    //getter
    const Vector3f & pos() const;
    const Vector3f & target() const;
    Vector3f spotDirection() const;

    float fov() const; //in degrees
    float spotMin() const;
    float spotMax() const;
    

    //setter
    void setPos(const Vector3f & light_pos);
    void setTarget(const Vector3f & light_target);
    void setFov(float light_fov);  //in degrees
    void setSpotMin(float light_spot_min);
    void setSpotMax(float light_spot_max);

private:
    Vector3f pos_;
    Vector3f target_;
    float fov_ = 45.0f;
    float spot_min_ = 0.2f;
    float spot_max_ = 0.5f;
};

}