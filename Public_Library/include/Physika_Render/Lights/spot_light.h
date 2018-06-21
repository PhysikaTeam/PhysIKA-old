/*
 * @file spot_light.h 
 * @Brief class SpotLight
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

#include <glm/glm.hpp>

#include "Physika_Core/Utilities/math_utilities.h"
#include "point_light.h"

namespace Physika{

class SpotLight: public PointLight
{
public:
    SpotLight() = default;
    SpotLight(const Color4f & ambient_col, 
              const Color4f & diffuse_col, 
              const Color4f & specular_col,
              const Vector3f & pos,
              float constant_attenuation,
              float linear_attenuation,
              float quadratic_attenution,
              const Vector3f & spot_direction,
              float spot_exponent,
              float spot_cutoff);

    LightType type() const override;
    void configToCurBindShader(const std::string & light_str) override;

    glm::mat4 lightProjMatrix() const;
    glm::mat4 lightViewMatrix() const;
    glm::mat4 lightTransformMatrix() const;

    //Note: we define spotDirection() as virtual since we would change its definition is FlashLight Class
    void setSpotDirection(const Vector3f & direction); 
    virtual const Vector3f spotDirection() const; //return by value

    void setSpotExponent(float exponent);
    float spotExponent() const;

    void setSpotCutoff(float cutoff);            //in degrees
    float spotCutoff() const;                    //in degrees

    //Note: Both spot_exponent & spot_outer_cutoff are used to implement the smooth/soft spot light effect
    void setSpotOuterCutoff(float outer_cutoff); //in degrees
    float spotOuterCutoff() const;               //in degrees


private:
    Vector3f spot_direction_ = {0.0f, 0.0f, -1.0f};
    float spot_exponent_ = 0.0f;
    float spot_cutoff_ = 180.0f;     //in degrees
    float spot_outer_cutoff_ = 0.0f; //in degrees, 0 means no using of outer_cutoff
};

} //end of namespace Physika

