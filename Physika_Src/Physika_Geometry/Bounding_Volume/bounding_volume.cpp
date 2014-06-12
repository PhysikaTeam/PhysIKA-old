/*
 * @file  bounding_volume.cpp
 * @bounding volume of a collidable object
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

#include "Physika_Geometry/Bounding_Volume/bounding_volume.h"

namespace Physika{

template <typename Scalar, int Dim>
BoundingVolume<Scalar, Dim>::BoundingVolume()
{
}

template <typename Scalar, int Dim>
BoundingVolume<Scalar, Dim>::~BoundingVolume()
{
}

//explicit instantitation
template class BoundingVolume<float, 3>;
template class BoundingVolume<double, 3>;

}