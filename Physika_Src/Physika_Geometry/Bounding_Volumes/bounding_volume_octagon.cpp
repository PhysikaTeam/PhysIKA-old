/*
 * @file  bounding_volume_octagon.cpp
 * @octagon bounding volume of a 2D collidable object
 * @author Tianxiang Zhang, Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <float.h>
#include <iostream>
#include "Physika_Geometry/Bounding_Volumes/bounding_volume_octagon.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Utilities/physika_exception.h"

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
    throw PhysikaException("Not implemented!");
}

template <typename Scalar>
void BoundingVolumeOctagon<Scalar>::setBoundingVolume(const Vector<Scalar, 2>& point)
{
    throw PhysikaException("Not implemented!");
}


template <typename Scalar>
void BoundingVolumeOctagon<Scalar>::setBoundingVolume(const Vector<Scalar, 2>& point_a, const Vector<Scalar, 2>& point_b)
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar>
typename BoundingVolumeInternal::BVType BoundingVolumeOctagon<Scalar>::bvType() const
{
    return BoundingVolumeInternal::OCTAGON;
}

template <typename Scalar>
bool BoundingVolumeOctagon<Scalar>::isOverlap(const BoundingVolume<Scalar, 2>* const bounding_volume) const
{
    throw PhysikaException("Not implemented!");
    return false;
}

template <typename Scalar>
bool BoundingVolumeOctagon<Scalar>::isOverlap(const BoundingVolume<Scalar, 2>* const bounding_volume, BoundingVolume<Scalar, 2>* return_volume) const
{
    throw PhysikaException("Not implemented!");
    return false;
}

template <typename Scalar>
bool BoundingVolumeOctagon<Scalar>::isInside(const Vector<Scalar, 2> &point) const
{
    throw PhysikaException("Not implemented!");
    return false;
}

template <typename Scalar>
Vector<Scalar, 2> BoundingVolumeOctagon<Scalar>::center() const
{
    throw PhysikaException("Not implemented!");
    return Vector<Scalar, 2>(0, 0);
}

template <typename Scalar>
Scalar BoundingVolumeOctagon<Scalar>::width() const
{
    throw PhysikaException("Not implemented!");
    return 0;
}

template <typename Scalar>
Scalar BoundingVolumeOctagon<Scalar>::height() const
{
    throw PhysikaException("Not implemented!");
    return 0;
}

template <typename Scalar>
Scalar BoundingVolumeOctagon<Scalar>::depth() const
{
    throw PhysikaException("Not implemented!");
    return 0;
}

template <typename Scalar>
void BoundingVolumeOctagon<Scalar>::setEmpty()
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar>
void BoundingVolumeOctagon<Scalar>::unionWith(const Vector<Scalar,2> &point)
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar>
void BoundingVolumeOctagon<Scalar>::unionWith(const BoundingVolume<Scalar,2>* const bounding_volume)
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar>
void BoundingVolumeOctagon<Scalar>::obtainUnion(const BoundingVolume<Scalar,2>* const bounding_volume_lhs, const BoundingVolume<Scalar,2>* const bounding_volume_rhs)
{
    throw PhysikaException("Not implemented!");
}

//explicit instantitation
template class BoundingVolumeOctagon<float>;
template class BoundingVolumeOctagon<double>;

}