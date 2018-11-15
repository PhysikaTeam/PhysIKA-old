/*
 * @file color_bar.cpp
 * @Brief a color bar class for OpenGL.
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

#include "Physika_Render/ColorBar/color_bar.h"

namespace Physika{

template <typename Scalar>
ColorBar<Scalar>::ColorBar()
    :width_(20),height_(300),start_point_(10,70),enable_horizon_(false)
{

}

template <typename Scalar>
ColorBar<Scalar>::~ColorBar()
{

}

template <typename Scalar>
const Vector<Scalar, 2> & ColorBar<Scalar>::startPoint() const
{
    return this->start_point_;
}

template <typename Scalar>
unsigned int ColorBar<Scalar>::width() const
{
    return this->width_;
}

template <typename Scalar>
unsigned int ColorBar<Scalar>::height() const
{
    return this->height_;
}

template <typename Scalar>
const ColorMap<Scalar> & ColorBar<Scalar>::colorMap() const
{
    return this->color_map_;
}

template <typename Scalar>
void ColorBar<Scalar>::setColorMap(const ColorMap<Scalar> & color_map)
{
    this->color_map_ = color_map_;
}

template <typename Scalar>
void ColorBar<Scalar>::setColorMapSize(unsigned int color_size)
{
    this->color_map_.setColorMapSize(color_size);
}

template <typename Scalar>
void ColorBar<Scalar>::setColorMapType(ColorMapType color_map_type)
{
    this->color_map_.setColorMapType(color_map_type);
}

template <typename Scalar>
void ColorBar<Scalar>::setColorMapTypeAndSize(ColorMapType color_map_type, unsigned int color_size)
{
    this->color_map_.setColorMapTypeAndSize(color_map_type, color_size);
}

template <typename Scalar>
void ColorBar<Scalar>::setStartPoint(const Vector<Scalar, 2> & start_point)
{
    this->start_point_ = start_point;
}

template <typename Scalar>
void ColorBar<Scalar>::setWidth(unsigned int width)
{
    this->width_ = width;
}

template <typename Scalar>
void ColorBar<Scalar>::setHeight(unsigned int height)
{
    this->height_ = height;
}

template <typename Scalar>
void ColorBar<Scalar>::setWidthAndHeight(unsigned int width, unsigned int height)
{
    this->width_ = width;
    this->height_ = height;
}

template <typename Scalar>
void ColorBar<Scalar>::enableHorizon()
{
    this->enable_horizon_ = true;
}

template <typename Scalar>
void ColorBar<Scalar>::disableHorizon()
{
    this->enable_horizon_ = false;
}

template <typename Scalar>
bool ColorBar<Scalar>::isHorizon() const
{
    return this->enable_horizon_;
}

//explicit instantiations
template class ColorBar<float>;
template class ColorBar<double>;

}