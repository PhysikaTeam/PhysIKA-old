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

template <typename Scalar,int Dim> class BoundingVolume;

namespace BoundingVolumeInternal{
    enum BVType {KDOP18, OCTAGON, AXIS_ALIGNED_BOX};

    template<typename Scalar, int Dim>
    BoundingVolume<Scalar, Dim>* createBoundingVolume(BVType bv_type);
}

template <typename Scalar,int Dim> class Vector;

template <typename Scalar,int Dim>
class BoundingVolume
{
public:
	//constructors && deconstructors
	BoundingVolume();
	virtual ~BoundingVolume();

	//set
	virtual void setBoundingVolume(const BoundingVolume<Scalar, Dim>* const bounding_volume) = 0;
	virtual void setBoundingVolume(const Vector<Scalar,Dim>& point) = 0;
	virtual void setBoundingVolume(const Vector<Scalar,Dim>& point_a, const Vector<Scalar,Dim>& point_b) = 0;

	virtual BoundingVolumeInternal::BVType bvType() const=0;

	//basic operation
	virtual bool isOverlap(const BoundingVolume<Scalar, Dim>* const bounding_volume) const = 0;
	virtual bool isOverlap(const BoundingVolume<Scalar, Dim>* const bounding_volume, BoundingVolume<Scalar, Dim>* return_volume) const = 0;
	virtual bool isInside(const Vector<Scalar,Dim> &point) const = 0;
	virtual Vector<Scalar,Dim> center() const = 0;
	//x-axis 
	virtual Scalar width() const = 0;
	//y-axis
	virtual Scalar height() const = 0;
	//z-axis
	virtual Scalar depth() const = 0;

	//*****WARNING! This function is used for setting a BV empty. For now it only has implement for float and double.*****
	//*****To support more types, add corresponding functions in the child class*****
	virtual void setEmpty() = 0;

	//union operation
	virtual void unionWith(const Vector<Scalar,Dim> &point) = 0;
	virtual void unionWith(const BoundingVolume<Scalar, Dim>* const bounding_volume) = 0;
	virtual void obtainUnion(const BoundingVolume<Scalar, Dim>* const bounding_volume_lhs, const BoundingVolume<Scalar, Dim>* const bounding_volume_rhs) = 0;

protected:
	
};

}  //end of namespace Physika

#endif  //PHYSIKA_GEOMETRY_BOUNDING_VOLUME_BOUNDING_VOLUME_H_