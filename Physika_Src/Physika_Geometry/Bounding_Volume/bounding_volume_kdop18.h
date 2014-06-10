/*
 * @file  bounding_volume_kdop18.h
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

#ifndef PHYSIKA_GEOMETRY_BOUNDING_VOLUME_BOUNDING_VOLUME_KDOP18_H_
#define PHYSIKA_GEOMETRY_BOUNDING_VOLUME_BOUNDING_VOLUME_KDOP18_H_

namespace Physika{

template <typename Scalar, int Dim> class Vector;
template <typename Scalar, int Dim> class BoundingVolume;

template <typename Scalar, int Dim>
class BoundingVolumeKDOP18 : public BoundingVolume<Scalar, Dim>
{
public:
	//constructors && deconstructors
	BoundingVolumeKDOP18();
	~BoundingVolumeKDOP18();

	//set
	inline void setBoundingVolume(const Vector<Scalar, Dim>& point);
	inline void setBoundingVolume(const Vector<Scalar, Dim>& point_a, const Vector<Scalar, Dim>& point_b);

	//basic check
	inline bool isOverlap(const BoundingVolumeKDOP18<Scalar, Dim>& bounding_volume) const;
	inline bool isOverlap(const BoundingVolumeKDOP18<Scalar, Dim>& bounding_volume, BoundingVolumeKDOP18<Scalar, Dim>& return_volume) const;
	inline bool isInside(const Vector<Scalar,Dim> &point) const;

	//union operation
	inline void unionWith(const BoundingVolumeKDOP18& bounding_volume);
	inline void obtainUnion(const BoundingVolumeKDOP18& bounding_volume_lhs, const BoundingVolumeKDOP18& bounding_volume_rhs);

protected:
	Scalar dist_[18];
};

}  //end of namespace Physika

#endif  //PHYSIKA_GEOMETRY_BOUNDING_VOLUME_BOUNDING_VOLUME_KDOP18_H_