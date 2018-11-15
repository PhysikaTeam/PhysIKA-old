/*
 * @file point_light.h 
 * @brief class PointLight.
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

#include "light_base.h"

namespace Physika{

class PointLight: public LightBase
{
public:
    PointLight() = default;
    PointLight(const Color4f & ambient_col, 
               const Color4f & diffuse_col, 
               const Color4f & specular_col,
               const Vector3f & pos,
               float constant_attenuation,
               float linear_attenuation,
               float quadratic_attenution);

    LightType type() const override;
    void configToCurBindShader(const std::string & light_str) override;
    
    void setPos(const Vector3f & pos); 

    //Note: we define pos() as virtual since we would change its definition is FlashLight Class
    virtual const Vector3f pos() const; //return by value

    void setConstantAttenuation(float constant_atten);
    float constantAttenuation() const;

    void setLinearAttenuation(float linear_atten);
    float linearAttenuation() const;

    void setQuadraticAttenuation(float quad_atten);
    float quadraticAttenuation() const;

private:
    Vector3f pos_ = { 0.0f, 0.0f, 1.0f };
    float constant_attenuation_ = 1.0f;
    float linear_attenuation_ = 0.0f;
    float quadratic_attenution_ = 0.0f;
};
    
}//end of namespace Physika