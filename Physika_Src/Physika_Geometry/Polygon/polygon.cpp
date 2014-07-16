/*
 * @file  polygon.cpp
 * @class of 2D polygon
 * @author Tianxiang Zhang
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Geometry/Polygon/polygon.h"

namespace Physika{

template<typename Scalar>
Polygon<Scalar>::Polygon()
{

}

template<typename Scalar>
Polygon<Scalar>::~Polygon()
{

}

//explicit instantitation
template class Polygon<float>;
template class Polygon<double>;

}