/*
 * @file  bounding_volume_kdop18.cpp
 * @18-DOP bounding volume of a collidable object
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

#include "Physika_Geometry\Bounding_Volume\bounding_volume_kdop18.h"
#include "Physika_Geometry\Bounding_Volume\bounding_volume.h"

namespace Physika{

template <typename Scalar, int Dim>
BoundingVolumeKDOP18<Scalar, Dim>::BoundingVolumeKDOP18()
{
}

template <typename Scalar, int Dim>
BoundingVolumeKDOP18<Scalar, Dim>::~BoundingVolumeKDOP18()
{
}

template <typename Scalar, int Dim>
inline void BoundingVolumeKDOP18<Scalar, Dim>::setBoundingVolume(const Vector<Scalar, Dim>& point)
{
}

template <typename Scalar, int Dim>
inline void BoundingVolumeKDOP18<Scalar, Dim>::setBoundingVolume(const Vector<Scalar, Dim>& point_a, const Vector<Scalar, Dim>& point_b)
{
}

template <typename Scalar, int Dim>
inline bool BoundingVolumeKDOP18<Scalar, Dim>::isOverlap(const BoundingVolumeKDOP18<Scalar, Dim>& bounding_volume) const
{
	return false;
}

template <typename Scalar, int Dim>
inline bool BoundingVolumeKDOP18<Scalar, Dim>::isOverlap(const BoundingVolumeKDOP18<Scalar, Dim>& bounding_volume, BoundingVolumeKDOP18<Scalar, Dim>& return_volume) const
{
	return false;
}

template <typename Scalar, int Dim>
inline bool BoundingVolumeKDOP18<Scalar, Dim>::isInside(const Vector<Scalar,Dim>& point) const
{
	return false;
}

template <typename Scalar, int Dim>
inline void BoundingVolumeKDOP18<Scalar, Dim>::unionWith (const BoundingVolumeKDOP18& bounding_volume)
{
}

template <typename Scalar, int Dim>
inline void BoundingVolumeKDOP18<Scalar, Dim>::obtainUnion(const BoundingVolumeKDOP18& bounding_volume_lhs, const BoundingVolumeKDOP18& bounding_volume_rhs)
{
}



//explicit instantitation
template class BoundingVolumeKDOP18<float, 3>;
template class BoundingVolumeKDOP18<double, 3>;

}