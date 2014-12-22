/*
 * @file  bounding_volume_AxisAlignedBox.h
 * @Axial Box bounding volume of a collidable object
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef	PHYSIKA_GEMETRY_BOUNDING_VOLUME_BOUNDING_VOLUME_AXISALIGNEDBOX
#define PHYSIKA_GEMETRY_BOUNDING_VOLUME_BOUNDING_VOLUME_AXISALIGNEDBOX

#include "Physika_Geometry/Bounding_Volume/bounding_volume.h"

namespace Physika{
	
template <typename Scalar, int Dim> class Vector;

template <typename Scalar>
class BoundingVolumeAxisAlignedBox : public BoundingVolume<Scalar, 3>
{
public:
	// constructor && destructor
	BoundingVolumeAxisAlignedBox();
	~BoundingVolumeAxisAlignedBox();

	// set
	virtual void setBoundingVolume(const BoundingVolume<Scalar, 3>* const bounding_volume);
	virtual void setBoundingVolume(const Vector<Scalar, 3>& point);
	virtual void setBoundingVolume(const Vector<Scalar,3>& point_a, const Vector<Scalar,3>& point_b);

	typename BoundingVolumeInternal::BVType bvType() const;

	//basic operation
	bool isOverlap(const BoundingVolume<Scalar, 3>* const bounding_volume) const;
	bool isOverlap(const BoundingVolume<Scalar, 3>* const bounding_volume, BoundingVolume<Scalar, 3>* return_volume) const;
	bool isInside(const Vector<Scalar, 3> &point) const;
	Vector<Scalar, 3> center() const;
	Scalar width() const;
	Scalar height() const;
	Scalar depth() const;
	void setEmpty();

	//union operation
	void unionWith(const Vector<Scalar, 3> &point);
	void unionWith(const BoundingVolume<Scalar, 3>* const bounding_volume);
	void obtainUnion(const BoundingVolume<Scalar, 3>* const bounding_volume_lhs, const BoundingVolume<Scalar, 3>* const bounding_volume_rhs);

protected:
	//faces of a axial box
	Scalar dist_[6];
};

} // end of namespace Physika
#endif // PHYSIKA_GEMETRY_BOUNDING_VOLUME_BOUNDING_VOLUME_AXISALIGNEDBOX