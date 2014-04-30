/*
 * @file point_render.cpp 
 * @Brief render of point, it is used to draw the simulation result of points.
 * @author Sheng Yang, Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Core/Array/array.h"
#include "Physika_Render/Point_Render/point_render.h"

namespace Physika{

template <typename PointType>
const unsigned int PointRender<PointType>::default_point_size_ = 2;

template <typename PointType>
PointRender<PointType>::PointRender():points_(NULL)
{
    point_size_ = default_point_size_;
}

template <typename PointType>
PointRender<PointType>::PointRender(const Array<PointType> *points):points_(points)
{
    point_size_ = default_point_size_;
}

template <typename PointType>
PointRender<PointType>::~PointRender(void)
{
}

template <typename PointType>
void PointRender<PointType>::render()
{
//TO DO: render point data
}

} //end of namespace Physika

