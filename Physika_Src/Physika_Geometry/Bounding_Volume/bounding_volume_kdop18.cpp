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


#include <float.h>
#include "Physika_Geometry/Bounding_Volume/bounding_volume_kdop18.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Utilities/math_utilities.h"

namespace Physika{

template <typename Scalar, int Dim>
BoundingVolumeKDOP18<Scalar, Dim>::BoundingVolumeKDOP18()
{
	setEmpty();
}

template <typename Scalar, int Dim>
BoundingVolumeKDOP18<Scalar, Dim>::~BoundingVolumeKDOP18()
{
}

template <typename Scalar, int Dim>
void BoundingVolumeKDOP18<Scalar, Dim>::getDistances(const Vector<Scalar, Dim>& point, Scalar& d3, Scalar& d4, Scalar& d5, Scalar& d6, Scalar& d7, Scalar& d8) const
{
	d3 = point[0] + point[1];
	d4 = point[0] + point[2];
	d5 = point[1] + point[2];
	d6 = point[0] - point[1];
	d7 = point[0] - point[2];
	d8 = point[1] - point[2];
}

template <typename Scalar, int Dim>
void BoundingVolumeKDOP18<Scalar, Dim>::getDistances(const Vector<Scalar, Dim>& point, Scalar d[]) const
{
	d[0] = point[0] + point[1];
	d[1] = point[0] + point[2];
	d[2] = point[1] + point[2];
	d[3] = point[0] - point[1];
	d[4] = point[0] - point[2];
	d[5] = point[1] - point[2];
}

template <typename Scalar, int Dim>
void BoundingVolumeKDOP18<Scalar, Dim>::setBoundingVolume(const BoundingVolume<Scalar, Dim>* const bounding_volume)
{
	if(bounding_volume->getBVType() != getBVType())
		return;
	for(int i = 0; i < 9; ++i)
	{
		dist_[i] = ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[i];
	}
}

template <typename Scalar, int Dim>
void BoundingVolumeKDOP18<Scalar, Dim>::setBoundingVolume(const Vector<Scalar, Dim>& point)
{
	dist_[0] = dist_[9]  = point[0];
	dist_[1] = dist_[10] = point[1];
	dist_[2] = dist_[11] = point[2];

	Scalar d3, d4, d5, d6, d7, d8;
	getDistances(point, d3, d4, d5, d6, d7, d8);
	dist_[3] = dist_[12] = d3;
	dist_[4] = dist_[13] = d4;
	dist_[5] = dist_[14] = d5;
	dist_[6] = dist_[15] = d6;
	dist_[7] = dist_[16] = d7;
	dist_[8] = dist_[17] = d8;
}

template <typename Scalar, int Dim>
void BoundingVolumeKDOP18<Scalar, Dim>::setBoundingVolume(const Vector<Scalar, Dim>& point_a, const Vector<Scalar, Dim>& point_b)
{
		dist_[0]  = min(point_a[0], point_b[0]);
		dist_[9]  = max(point_a[0], point_b[0]);
		dist_[1]  = min(point_a[1], point_b[1]);
		dist_[10] = max(point_a[1], point_b[1]);
		dist_[2]  = min(point_a[2], point_b[2]);
		dist_[11] = max(point_a[2], point_b[2]);

		Scalar ad3, ad4, ad5, ad6, ad7, ad8;
		getDistances(point_a, ad3, ad4, ad5, ad6, ad7, ad8);
		Scalar bd3, bd4, bd5, bd6, bd7, bd8;
		getDistances(point_b, bd3, bd4, bd5, bd6, bd7, bd8);
		dist_[3]  = min(ad3, bd3);
		dist_[12] = max(ad3, bd3);
		dist_[4]  = min(ad4, bd4);
		dist_[13] = max(ad4, bd4);
		dist_[5]  = min(ad5, bd5);
		dist_[14] = max(ad5, bd5);
		dist_[6]  = min(ad6, bd6);
		dist_[15] = max(ad6, bd6);
		dist_[7]  = min(ad7, bd7);
		dist_[16] = max(ad7, bd7);
		dist_[8]  = min(ad8, bd8);
		dist_[17] = max(ad8, bd8);
}

template <typename Scalar, int Dim>
typename BoundingVolume<Scalar, Dim>::BVType BoundingVolumeKDOP18<Scalar, Dim>::getBVType() const
{
	return BoundingVolume<Scalar,Dim>::KDOP18;
}

template <typename Scalar, int Dim>
bool BoundingVolumeKDOP18<Scalar, Dim>::isOverlap(const BoundingVolume<Scalar, Dim>* const bounding_volume) const
{
	if(bounding_volume->getBVType() != getBVType())
		return false;
	for (int i = 0; i < 9; ++i)
	{
		if (dist_[i] > ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[i+9])
			return false;
		if (dist_[i+9] < ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[i])
			return false;
	}
	return true;
}

template <typename Scalar, int Dim>
bool BoundingVolumeKDOP18<Scalar, Dim>::isOverlap(const BoundingVolume<Scalar, Dim>* const bounding_volume, BoundingVolume<Scalar, Dim>* return_volume) const
{
	if(bounding_volume->getBVType() != getBVType())
		return false;
	if(return_volume->getBVType() != getBVType())
		return false;
	if (!isOverlap(bounding_volume))
		return false;
	for (int i = 0; i < 9; ++i)
	{
		((BoundingVolumeKDOP18<Scalar,Dim>*)return_volume)->dist_[i] = max(dist_[i],  ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[i]);
		((BoundingVolumeKDOP18<Scalar,Dim>*)return_volume)->dist_[i+9] = min(dist_[i+9], ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[i+9]);
	}
	return true;
}

template <typename Scalar, int Dim>
bool BoundingVolumeKDOP18<Scalar, Dim>::isInside(const Vector<Scalar,Dim>& point) const
{
	for (int i = 0; i < 3; ++i) 
	{
		if (point[i] < dist_[i] || point[i] > dist_[i+9])
			return false;
	}
	Scalar d[6];
	getDistances(point, d);
	for (int i = 3; i < 9; ++i) {
		if (d[i-3] < dist_[i] || d[i-3] > dist_[i+9])
			return false;
	}
	return true;
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> BoundingVolumeKDOP18<Scalar, Dim>::center() const
{
	Vector<Scalar,Dim> center;
	center[0] = (dist_[0]+dist_[9])/2;
	center[1] = (dist_[1]+dist_[10])/2;
	center[2] = (dist_[2]+dist_[11])/2;
	return center;
}

template <typename Scalar, int Dim>
Scalar BoundingVolumeKDOP18<Scalar, Dim>::width() const
{
	return dist_[9] - dist_[0];
}

template <typename Scalar, int Dim>
Scalar BoundingVolumeKDOP18<Scalar, Dim>::height() const
{
	return dist_[10] - dist_[1];
}

template <typename Scalar, int Dim>
Scalar BoundingVolumeKDOP18<Scalar, Dim>::depth() const
{
	return dist_[11] - dist_[2];
}

template <typename Scalar, int Dim>
void BoundingVolumeKDOP18<Scalar, Dim>::setEmpty()
{
	for (int i = 0; i < 9; ++i)
	{
		dist_[i] = FLT_MAX;
		dist_[i+9] = -FLT_MAX;
	}
}

template <typename Scalar, int Dim>
void BoundingVolumeKDOP18<Scalar, Dim>::unionWith (const Vector<Scalar,Dim> &point)
{
	dist_[0]  = min(point[0], dist_[0]);
	dist_[9]  = max(point[0], dist_[9]);
	dist_[1]  = min(point[1], dist_[1]);
	dist_[10] = max(point[1], dist_[10]);
	dist_[2]  = min(point[2], dist_[2]);
	dist_[11] = max(point[2], dist_[11]);

	Scalar d3, d4, d5, d6, d7, d8;
	getDistances(point, d3, d4, d5, d6, d7, d8);
	dist_[3]  = min(d3, dist_[3]);
	dist_[12] = max(d3, dist_[12]);
	dist_[4]  = min(d4, dist_[4]);
	dist_[13] = max(d4, dist_[13]);
	dist_[5]  = min(d5, dist_[5]);
	dist_[14] = max(d5, dist_[14]);
	dist_[6]  = min(d6, dist_[6]);
	dist_[15] = max(d6, dist_[15]);
	dist_[7]  = min(d7, dist_[7]);
	dist_[16] = max(d7, dist_[16]);
	dist_[8]  = min(d8, dist_[8]);
	dist_[17] = max(d8, dist_[17]);
}

template <typename Scalar, int Dim>
void BoundingVolumeKDOP18<Scalar, Dim>::unionWith (const BoundingVolume<Scalar,Dim>* const bounding_volume)
{
	if(bounding_volume->getBVType() != getBVType())
		return;

	dist_[0]  = min(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[0], dist_[0]);
	dist_[9]  = max(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[9], dist_[9]);
	dist_[1]  = min(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[1], dist_[1]);
	dist_[10] = max(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[10], dist_[10]);
	dist_[2]  = min(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[2], dist_[2]);
	dist_[11] = max(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[11], dist_[11]);
	dist_[3]  = min(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[3], dist_[3]);
	dist_[12] = max(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[12], dist_[12]);
	dist_[4]  = min(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[4], dist_[4]);
	dist_[13] = max(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[13], dist_[13]);
	dist_[5]  = min(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[5], dist_[5]);
	dist_[14] = max(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[14], dist_[14]);
	dist_[6]  = min(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[6], dist_[6]);
	dist_[15] = max(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[15], dist_[15]);
	dist_[7]  = min(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[7], dist_[7]);
	dist_[16] = max(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[16], dist_[16]);
	dist_[8]  = min(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[8], dist_[8]);
	dist_[17] = max(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume)->dist_[17], dist_[17]);
}

template <typename Scalar, int Dim>
void BoundingVolumeKDOP18<Scalar, Dim>::obtainUnion(const BoundingVolume<Scalar,Dim>* const bounding_volume_lhs, const BoundingVolume<Scalar,Dim>* const bounding_volume_rhs)
{
	if(bounding_volume_lhs->getBVType() != getBVType())
		return;
	if(bounding_volume_rhs->getBVType() != getBVType())
		return;

	dist_[0]  = min(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_lhs)->dist_[0], ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_rhs)->dist_[0]);
	dist_[9]  = max(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_lhs)->dist_[9], ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_rhs)->dist_[9]);
	dist_[1]  = min(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_lhs)->dist_[1], ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_rhs)->dist_[1]);
	dist_[10] = max(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_lhs)->dist_[10], ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_rhs)->dist_[10]);
	dist_[2]  = min(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_lhs)->dist_[2], ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_rhs)->dist_[2]);
	dist_[11] = max(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_lhs)->dist_[11], ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_rhs)->dist_[11]);
	dist_[3]  = min(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_lhs)->dist_[3], ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_rhs)->dist_[3]);
	dist_[12] = max(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_lhs)->dist_[12], ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_rhs)->dist_[12]);
	dist_[4]  = min(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_lhs)->dist_[4], ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_rhs)->dist_[4]);
	dist_[13] = max(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_lhs)->dist_[13], ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_rhs)->dist_[13]);
	dist_[5]  = min(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_lhs)->dist_[5], ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_rhs)->dist_[5]);
	dist_[14] = max(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_lhs)->dist_[14], ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_rhs)->dist_[14]);
	dist_[6]  = min(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_lhs)->dist_[6], ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_rhs)->dist_[6]);
	dist_[15] = max(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_lhs)->dist_[15], ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_rhs)->dist_[15]);
	dist_[7]  = min(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_lhs)->dist_[7], ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_rhs)->dist_[7]);
	dist_[16] = max(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_lhs)->dist_[16], ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_rhs)->dist_[16]);
	dist_[8]  = min(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_lhs)->dist_[8], ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_rhs)->dist_[8]);
	dist_[17] = max(((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_lhs)->dist_[17], ((BoundingVolumeKDOP18<Scalar,Dim>*)bounding_volume_rhs)->dist_[17]);
}

//explicit instantitation
template class BoundingVolumeKDOP18<float, 3>;
template class BoundingVolumeKDOP18<double, 3>;

}
