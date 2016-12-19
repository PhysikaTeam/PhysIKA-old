/*
 * @file  bounding_volume_axis_aligned_box.cpp
 * @Axial Box bounding volume of a collidable object
 * @author Wei Chen, Fei Zhu
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
#include "Physika_Geometry/Bounding_Volumes/bounding_volume_axis_aligned_box.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Utilities/math_utilities.h"

namespace Physika{

template <typename Scalar>
BoundingVolumeAxisAlignedBox<Scalar>::BoundingVolumeAxisAlignedBox()
{
    setEmpty();
}

template <typename Scalar>
BoundingVolumeAxisAlignedBox<Scalar>::~BoundingVolumeAxisAlignedBox()
{
    
}

template <typename Scalar>
void BoundingVolumeAxisAlignedBox<Scalar>::setBoundingVolume(const BoundingVolume<Scalar, 3>* const bounding_volume)
{
    if (bounding_volume->bvType() != bvType())
    {
        return;
    }
    for (int i=0; i<6; i++)
    {
        dist_[i] = ((BoundingVolumeAxisAlignedBox<Scalar>*)bounding_volume)->dist_[i];
    }
}

template <typename Scalar>
void BoundingVolumeAxisAlignedBox<Scalar>::setBoundingVolume(const Vector<Scalar, 3>& point)
{
    // need further consideration
    dist_[0] = dist_[3] = point[0];
    dist_[1] = dist_[4] = point[1];
    dist_[2] = dist_[5] = point[2];
}

template <typename Scalar>
void BoundingVolumeAxisAlignedBox<Scalar>::setBoundingVolume(const Vector<Scalar,3>& point_a, const Vector<Scalar,3>& point_b)
{
    if (point_a[0] < point_b[0])
    {
        dist_[0] = point_a[0];
        dist_[3] = point_b[0];
    }
    else
    {
        dist_[0] = point_b[0];
        dist_[3] = point_a[0];
    }

    if (point_a[1] < point_b[1])
    {
        dist_[1] = point_a[1];
        dist_[4] = point_b[1];
    }
    else
    {
        dist_[1] = point_b[1];
        dist_[4] = point_a[1];
    }

    if (point_a[2] < point_b[2])
    {
        dist_[2] = point_a[2];
        dist_[5] = point_b[2];
    }
    else
    {
        dist_[2] = point_b[2];
        dist_[5] = point_a[2];
    }
}

template <typename Scalar>
typename BoundingVolumeInternal::BVType BoundingVolumeAxisAlignedBox<Scalar>::bvType() const
{
    return BoundingVolumeInternal::AXIS_ALIGNED_BOX;
}

template <typename Scalar>
bool BoundingVolumeAxisAlignedBox<Scalar>::isOverlap(const BoundingVolume<Scalar, 3>* const bounding_volume) const
{
    if(bounding_volume == NULL)
    {
        std::cerr<<"Null bounding!"<<std::endl;
        return false;
    }
    if(bounding_volume->bvType() != bvType())
    {
        std::cerr<<"Bounding volume type mismatch!"<<std::endl;
        return false;
    }
    const BoundingVolumeAxisAlignedBox<Scalar> * const bv = dynamic_cast<const BoundingVolumeAxisAlignedBox<Scalar>* const>(bounding_volume);
    PHYSIKA_ASSERT(bv);
    if (dist_[0]>bv->dist_[3] || dist_[3]<bv->dist_[0] || dist_[1]>bv->dist_[4] || dist_[4]<bv->dist_[1] || dist_[2]>bv->dist_[5] || dist_[5]<bv->dist_[2])
    {
        return false;
    }
    return true;
}

template <typename Scalar>
bool BoundingVolumeAxisAlignedBox<Scalar>::isOverlap(const BoundingVolume<Scalar, 3>* const bounding_volume, BoundingVolume<Scalar, 3>* return_volume) const
{
    if(bounding_volume == NULL || return_volume == NULL)
    {
        std::cerr<<"Null bounding!"<<std::endl;
        return false;
    }
    if(bounding_volume->bvType() != bvType())
    {
        std::cerr<<"Bounding volume type mismatch!"<<std::endl;
        return false;
    }
    if(return_volume->bvType() != bvType())
    {
        std::cerr<<"Bounding volume type mismatch!"<<std::endl;
        return false;
    }
    if (!isOverlap(bounding_volume))
        return false;

    const BoundingVolumeAxisAlignedBox<Scalar> * const bv = dynamic_cast<const BoundingVolumeAxisAlignedBox<Scalar>* const>(bounding_volume);
    BoundingVolumeAxisAlignedBox<Scalar> * return_bv = dynamic_cast<BoundingVolumeAxisAlignedBox<Scalar>*>(return_volume);
    PHYSIKA_ASSERT(bv);
    PHYSIKA_ASSERT(return_bv);
    return_bv->dist_[0] = min(bv->dist_[0],dist_[0]);
    return_bv->dist_[1] = min(bv->dist_[1],dist_[1]);
    return_bv->dist_[2] = min(bv->dist_[2],dist_[2]);

    return_bv->dist_[3] = max(bv->dist_[3],dist_[3]);
    return_bv->dist_[4] = max(bv->dist_[4],dist_[4]);
    return_bv->dist_[5] = max(bv->dist_[5],dist_[5]);
}

template <typename Scalar>
bool BoundingVolumeAxisAlignedBox<Scalar>::isInside(const Vector<Scalar, 3> &point) const
{
    if (point[0]>=dist_[0] && point[0]<=dist_[3] && point[1]>=dist_[1] && point[1]<=dist_[4] && point[2]>=dist_[2] && point[2]<=dist_[5])
    {
        return true;
    }
    return false;
}

template <typename Scalar>
Vector<Scalar,3> BoundingVolumeAxisAlignedBox<Scalar>::center() const
{
    Vector<Scalar, 3> center;
    center[0] = (dist_[0]+dist_[3])/2;
    center[1] = (dist_[1]+dist_[4])/2;
    center[2] = (dist_[2]+dist_[5])/2;
    return center;
}

template <typename Scalar>
Scalar BoundingVolumeAxisAlignedBox<Scalar>::width() const
{
    return dist_[3]-dist_[0];
}

template <typename Scalar>
Scalar BoundingVolumeAxisAlignedBox<Scalar>::height() const
{
    return dist_[4]-dist_[1];
}

template <typename Scalar>
Scalar BoundingVolumeAxisAlignedBox<Scalar>::depth() const
{
    return dist_[5]-dist_[2];
}

template <typename Scalar>
void BoundingVolumeAxisAlignedBox<Scalar>::setEmpty()
{
    for (int i=0; i<3; i++)
    {
        dist_[i] = FLT_MAX;
        dist_[i+3] = -FLT_MAX;
    }
}
template <typename Scalar>
void BoundingVolumeAxisAlignedBox<Scalar>::unionWith(const Vector<Scalar, 3> &point)
{
    dist_[0] = min(point[0],dist_[0]);
    dist_[1] = min(point[1],dist_[1]);
    dist_[2] = min(point[2],dist_[2]);

    dist_[3] = max(point[0],dist_[3]);
    dist_[4] = max(point[1],dist_[4]);
    dist_[5] = max(point[2],dist_[5]);
}

template <typename Scalar>
void BoundingVolumeAxisAlignedBox<Scalar>::unionWith(const BoundingVolume<Scalar, 3>* const bounding_volume)
{
    if(bounding_volume == NULL)
    {
        std::cerr<<"Null bounding!"<<std::endl;
        return;
    }
    if(bounding_volume->bvType() != bvType())
    {
        std::cerr<<"Bounding volume type mismatch!"<<std::endl;
        return;
    }
    const BoundingVolumeAxisAlignedBox<Scalar> * const bv = dynamic_cast<const BoundingVolumeAxisAlignedBox<Scalar>* const>(bounding_volume);
    PHYSIKA_ASSERT(bv);
    dist_[0] = min(bv->dist_[0],dist_[0]);
    dist_[1] = min(bv->dist_[1],dist_[1]);
    dist_[2] = min(bv->dist_[2],dist_[2]);

    dist_[3] = max(bv->dist_[3],dist_[3]);
    dist_[4] = max(bv->dist_[4],dist_[4]);
    dist_[5] = max(bv->dist_[5],dist_[5]);
}

template <typename Scalar>
void BoundingVolumeAxisAlignedBox<Scalar>::obtainUnion(const BoundingVolume<Scalar, 3>* const bounding_volume_lhs, const BoundingVolume<Scalar, 3>* const bounding_volume_rhs)
{
    if(bounding_volume_lhs == NULL && bounding_volume_rhs == NULL)
    {
        std::cerr<<"Null bounding!"<<std::endl;
        return;
    }
    if(bounding_volume_lhs->bvType() != bvType())
    {
        std::cerr<<"Bounding volume type mismatch!"<<std::endl;
        return;
    }
    if(bounding_volume_rhs->bvType() != bvType())
    {
        std::cerr<<"Bounding volume type mismatch!"<<std::endl;
        return;
    }
    const BoundingVolumeAxisAlignedBox<Scalar> * const bv_lhs = dynamic_cast<const BoundingVolumeAxisAlignedBox<Scalar>* const>(bounding_volume_lhs);
    const BoundingVolumeAxisAlignedBox<Scalar> * const bv_rhs = dynamic_cast<const BoundingVolumeAxisAlignedBox<Scalar>* const>(bounding_volume_rhs);
    PHYSIKA_ASSERT(bv_lhs);
    PHYSIKA_ASSERT(bv_rhs);
    dist_[0] = min(bv_lhs->dist_[0],bv_rhs->dist_[0]);
    dist_[1] = min(bv_lhs->dist_[1],bv_rhs->dist_[1]);
    dist_[2] = min(bv_lhs->dist_[2],bv_rhs->dist_[2]);

    dist_[3] = max(bv_lhs->dist_[3],bv_rhs->dist_[3]);
    dist_[4] = max(bv_lhs->dist_[4],bv_rhs->dist_[4]);
    dist_[5] = max(bv_lhs->dist_[5],bv_rhs->dist_[5]);
}

// explicit instantiations 
template class BoundingVolumeAxisAlignedBox<float>;
template class BoundingVolumeAxisAlignedBox<double>;
}// end of namespace Physika