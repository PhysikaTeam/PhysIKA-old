/*
 * @file light-inl.h 
 * @Brief template function implemantation of  light class for OpenGL.
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

#ifndef PHYSIKA_RENDER_LIGHTS_LIGHT_INL_H_
#define PHYSIKA_RENDER_LIGHTS_LIGHT_INL_H_

namespace Physika{

template<typename ColorType>
void Light::setAmbient(const Color<ColorType>& color)
{
    openGLLightv(this->light_id_, GL_AMBIENT, color.template convertColor<float>());
}

template<typename ColorType>
Color<ColorType> Light::ambient() const
{
    float color[4];
    glGetLightfv(this->light_id_, GL_AMBIENT, color);
    Color<float> temp_color(color[0], color[1], color[2], color[3]);
    return temp_color.convertColor<ColorType>();
}

template<typename ColorType>
void Light::setDiffuse(const Color<ColorType>& color)
{
    openGLLightv(this->light_id_, GL_DIFFUSE, color.template convertColor<float>());
}

template<typename ColorType>
Color<ColorType> Light::diffuse() const
{
    float color[4];
    glGetLightfv(this->light_id_, GL_DIFFUSE, color);
    Color<float> temp_color(color[0], color[1], color[2], color[3]);
    return temp_color.convertColor<ColorType>();
}

template<typename ColorType>
void Light::setSpecular(const Color<ColorType>& color)
{
    openGLLightv(this->light_id_, GL_SPECULAR, color.template convertColor<float>());
}

template<typename ColorType>
Color<ColorType> Light::specular() const
{
    float color[4];
    glGetLightfv(this->light_id_, GL_SPECULAR, color);
    Color<float> temp_color(color[0], color[1], color[2], color[3]);
    return temp_color.convertColor<ColorType>();
}

template<typename Scalar>
void Light::setConstantAttenuation(Scalar const_atten)
{
    openGLLight(this->light_id_, GL_CONSTANT_ATTENUATION, const_atten);
}

template<typename Scalar>
Scalar Light::constantAttenuation() const
{
    float const_atten;
    glGetLightfv(this->light_id_, GL_CONSTANT_ATTENUATION, &const_atten );
    return static_cast<Scalar>(const_atten);
}

template<typename Scalar>
void Light::setLinearAttenuation(Scalar linear_atten)
{
    openGLLight(this->light_id_, GL_LINEAR_ATTENUATION, linear_atten);
}

template<typename Scalar>
Scalar Light::linearAttenuation() const
{
    float linear_atten;
    glGetLightfv(this->light_id_, GL_LINEAR_ATTENUATION, &linear_atten );
    return static_cast<Scalar>(linear_atten);
}

template<typename Scalar>
void Light::setQuadraticAttenuation(Scalar quad_atten)
{
    openGLLight(this->light_id_, GL_QUADRATIC_ATTENUATION, quad_atten);
}

template<typename Scalar>
Scalar Light::quadraticAttenuation() const
{
    float quad_atten;
    glGetLightfv(this->light_id_, GL_QUADRATIC_ATTENUATION, &quad_atten );
    return static_cast<Scalar>(quad_atten);
}

}// end of namespace Physika

#endif //PHYSIKA_RENDER_LIGHTS_LIGHT_INL_H_
