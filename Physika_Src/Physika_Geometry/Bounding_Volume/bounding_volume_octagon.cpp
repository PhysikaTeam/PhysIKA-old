/*
 * @file  bounding_volume_octagon.cpp
 * @octagon bounding volume of a 2D collidable object
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

#include <float.h>
#include "Physika_Geometry/Bounding_Volume/bounding_volume_octagon.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Utilities/math_utilities.h"

namespace Physika{

template <typename Scalar>
BoundingVolumeOctagon<Scalar>::BoundingVolumeOctagon()
{
    setEmpty();
}

template <typename Scalar>
BoundingVolumeOctagon<Scalar>::~BoundingVolumeOctagon()
{
}

template <typename Scalar>
void BoundingVolumeOctagon<Scalar>::setBoundingVolume(const BoundingVolume<Scalar, 2>* const bounding_volume)
{
    //to do
}

template <typename Scalar>
void BoundingVolumeOctagon<Scalar>::setBoundingVolume(const Vector<Scalar, 2>& point)
{
    //to do
}


template <typename Scalar>
void BoundingVolumeOctagon<Scalar>::setBoundingVolume(const Vector<Scalar, 2>& point_a, const Vector<Scalar, 2>& point_b)
{
    //to do
}

template <typename Scalar>
typename BoundingVolumeInternal::BVType BoundingVolumeOctagon<Scalar>::bvType() const
{
    return BoundingVolumeInternal::OCTAGON;
}

template <typename Scalar>
bool BoundingVolumeOctagon<Scalar>::isOverlap(const BoundingVolume<Scalar, 2>* const bounding_volume) const
{
    //to do
    return false;
}

template <typename Scalar>
bool BoundingVolumeOctagon<Scalar>::isOverlap(const BoundingVolume<Scalar, 2>* const bounding_volume, BoundingVolume<Scalar, 2>* return_volume) const
{
    //to do
    return false;
}

template <typename Scalar>
bool BoundingVolumeOctagon<Scalar>::isInside(const Vector<Scalar, 2> &point) const
{
    //to do
    return false;
}

template <typename Scalar>
Vector<Scalar, 2> BoundingVolumeOctagon<Scalar>::center() const
{
    //to do
    return Vector<Scalar, 2>(0, 0);
}

template <typename Scalar>
Scalar BoundingVolumeOctagon<Scalar>::width() const
{
    //to do
    return 0;
}

template <typename Scalar>
Scalar BoundingVolumeOctagon<Scalar>::height() const
{
    //to do
    return 0;
}

template <typename Scalar>
Scalar BoundingVolumeOctagon<Scalar>::depth() const
{
    //to do
    return 0;
}

template <typename Scalar>
void BoundingVolumeOctagon<Scalar>::setEmpty()
{
    //to do
}

template <typename Scalar>
void BoundingVolumeOctagon<Scalar>::unionWith(const Vector<Scalar,2> &point)
{
    //to do
}

template <typename Scalar>
void BoundingVolumeOctagon<Scalar>::unionWith(const BoundingVolume<Scalar,2>* const bounding_volume)
{
    //to do
}

template <typename Scalar>
void BoundingVolumeOctagon<Scalar>::obtainUnion(const BoundingVolume<Scalar,2>* const bounding_volume_lhs, const BoundingVolume<Scalar,2>* const bounding_volume_rhs)
{
    //to do
}

//explicit instantitation
template class BoundingVolumeOctagon<float>;
template class BoundingVolumeOctagon<double>;

}