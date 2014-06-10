/*
 * @file  bounding_volume.h
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

#ifndef PHYSIKA_GEOMETRY_BOUNDING_VOLUME_BOUNDING_VOLUME_H_
#define PHYSIKA_GEOMETRY_BOUNDING_VOLUME_BOUNDING_VOLUME_H_

namespace Physika{

template <typename Scalar,int Dim> class Vector;

template <typename Scalar,int Dim>
class BoundingVolume
{
public:
	//constructors && deconstructors
	BoundingVolume();
	virtual ~BoundingVolume();

	//set
	virtual inline void setBoundingVolume(const Vector<Scalar,Dim>& point) = 0;
	virtual inline void setBoundingVolume(const Vector<Scalar,Dim>& point_a, const Vector<Scalar,Dim>& point_b) = 0;

	//basic check
	virtual inline bool isOverlap(const BoundingVolume& bounding_volume) const = 0;
	virtual inline bool isOverlap(const BoundingVolume& bounding_volume, BoundingVolume& return_volume) const = 0;
	virtual inline bool isInside(const Vector<Scalar,Dim> &point) const = 0;

	//union operation
	virtual inline void unionWith(const BoundingVolume& bounding_volume) = 0;
	virtual inline void obtainUnion(const BoundingVolume& bounding_volume_lhs, const BoundingVolume& bounding_volume_rhs) = 0;

protected:
	
};

}  //end of namespace Physika

#endif  //PHYSIKA_GEOMETRY_BOUNDING_VOLUME_BOUNDING_VOLUME_H_