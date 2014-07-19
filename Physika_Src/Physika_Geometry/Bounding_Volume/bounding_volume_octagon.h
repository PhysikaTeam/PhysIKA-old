/*
 * @file  bounding_volume_octagon.h
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

#ifndef PHYSIKA_GEOMETRY_BOUNDING_VOLUME_BOUNDING_VOLUME_OCTAGON_H_
#define PHYSIKA_GEOMETRY_BOUNDING_VOLUME_BOUNDING_VOLUME_OCTAGON_H_

#include "Physika_Geometry/Bounding_Volume/bounding_volume.h"

namespace Physika{

template <typename Scalar, int Dim> class Vector;

template <typename Scalar>
class BoundingVolumeOctagon : public BoundingVolume<Scalar, 2>
{
public:
    //constructors && deconstructors
    BoundingVolumeOctagon();
    ~BoundingVolumeOctagon();

    //set
    void setBoundingVolume(const BoundingVolume<Scalar, 2>* const bounding_volume);
    void setBoundingVolume(const Vector<Scalar, 2>& point);
    void setBoundingVolume(const Vector<Scalar, 2>& point_a, const Vector<Scalar, 2>& point_b);
    typename BoundingVolumeInternal::BVType bvType() const;

    //basic operation
    bool isOverlap(const BoundingVolume<Scalar, 2>* const bounding_volume) const;
    bool isOverlap(const BoundingVolume<Scalar, 2>* const bounding_volume, BoundingVolume<Scalar, 2>* return_volume) const;
    bool isInside(const Vector<Scalar, 2> &point) const;
    Vector<Scalar, 2> center() const;
    Scalar width() const;
    Scalar height() const;
    Scalar depth() const;
    void setEmpty();

    //union operation
    void unionWith(const Vector<Scalar,2> &point);
    void unionWith(const BoundingVolume<Scalar,2>* const bounding_volume);
    void obtainUnion(const BoundingVolume<Scalar,2>* const bounding_volume_lhs, const BoundingVolume<Scalar,2>* const bounding_volume_rhs);

protected:
    //faces of a octagon
    Scalar dist_[8];

};

}

#endif //PHYSIKA_GEOMETRY_BOUNDING_VOLUME_BOUNDING_VOLUME_OCTAGON_H_