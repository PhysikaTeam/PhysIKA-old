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
void Light::setPosition(const Vector<Scalar,3>& pos)
{
    Vector<Scalar,4> position(pos[0], pos[1], pos[2], static_cast<Scalar>(1.0)); // the last one is 1.0 to specify this light is position based.
    openGLLightv(this->light_id_, GL_POSITION, position);
}

template<typename Scalar>
Vector<Scalar,3> Light::position() const
{
    float position[4];
    glGetLightfv(this->light_id_, GL_POSITION, position);
    return Vector<Scalar,3>(position[0], position[1], position[2]);
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
    return (Scalar) const_atten;
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
    return (Scalar) linear_atten;
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
    return (Scalar) quad_atten;
}

}// end of namespace Physika

#endif //PHYSIKA_RENDER_LIGHTS_LIGHT_INL_H_
