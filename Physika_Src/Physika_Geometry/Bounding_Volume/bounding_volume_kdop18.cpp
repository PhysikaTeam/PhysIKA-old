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
#include "Physika_Core/Vectors/vector_3d.h"

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
		dist_[i] = ((BoundingVolumeKDOP18*)bounding_volume)->dist_[i];
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
		dist_[0]  = MIN(point_a[0], point_b[0]);
		dist_[9]  = MAX(point_a[0], point_b[0]);
		dist_[1]  = MIN(point_a[1], point_b[1]);
		dist_[10] = MAX(point_a[1], point_b[1]);
		dist_[2]  = MIN(point_a[2], point_b[2]);
		dist_[11] = MAX(point_a[2], point_b[2]);

		Scalar ad3, ad4, ad5, ad6, ad7, ad8;
		getDistances(point_a, ad3, ad4, ad5, ad6, ad7, ad8);
		Scalar bd3, bd4, bd5, bd6, bd7, bd8;
		getDistances(point_b, bd3, bd4, bd5, bd6, bd7, bd8);
		dist_[3]  = MIN(ad3, bd3);
		dist_[12] = MAX(ad3, bd3);
		dist_[4]  = MIN(ad4, bd4);
		dist_[13] = MAX(ad4, bd4);
		dist_[5]  = MIN(ad5, bd5);
		dist_[14] = MAX(ad5, bd5);
		dist_[6]  = MIN(ad6, bd6);
		dist_[15] = MAX(ad6, bd6);
		dist_[7]  = MIN(ad7, bd7);
		dist_[16] = MAX(ad7, bd7);
		dist_[8]  = MIN(ad8, bd8);
		dist_[17] = MAX(ad8, bd8);
}

template <typename Scalar, int Dim>
typename BoundingVolume<Scalar, Dim>::BVType BoundingVolumeKDOP18<Scalar, Dim>::getBVType() const
{
	return KDOP18;
}

template <typename Scalar, int Dim>
bool BoundingVolumeKDOP18<Scalar, Dim>::isOverlap(const BoundingVolume<Scalar, Dim>* const bounding_volume) const
{
	if(bounding_volume->getBVType() != getBVType())
		return false;
	for (int i = 0; i < 9; ++i)
	{
		if (dist_[i] > ((BoundingVolumeKDOP18*)bounding_volume)->dist_[i+9])
			return false;
		if (dist_[i+9] < ((BoundingVolumeKDOP18*)bounding_volume)->dist_[i])
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
		((BoundingVolumeKDOP18*)return_volume)->dist_[i] = MAX(dist_[i],  ((BoundingVolumeKDOP18*)bounding_volume)->dist_[i]);
		((BoundingVolumeKDOP18*)return_volume)->dist_[i+9] = MIN(dist_[i+9], ((BoundingVolumeKDOP18*)bounding_volume)->dist_[i+9]);
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
void BoundingVolumeKDOP18<Scalar, Dim>::unionWith (const Vector<Scalar,Dim> &point)
{
	dist_[0]  = MIN(point[0], dist_[0]);
	dist_[9]  = MAX(point[0], dist_[9]);
	dist_[1]  = MIN(point[1], dist_[1]);
	dist_[10] = MAX(point[1], dist_[10]);
	dist_[2]  = MIN(point[2], dist_[2]);
	dist_[11] = MAX(point[2], dist_[11]);

	Scalar d3, d4, d5, d6, d7, d8;
	getDistances(point, d3, d4, d5, d6, d7, d8);
	dist_[3]  = MIN(d3, dist_[3]);
	dist_[12] = MAX(d3, dist_[12]);
	dist_[4]  = MIN(d4, dist_[4]);
	dist_[13] = MAX(d4, dist_[13]);
	dist_[5]  = MIN(d5, dist_[5]);
	dist_[14] = MAX(d5, dist_[14]);
	dist_[6]  = MIN(d6, dist_[6]);
	dist_[15] = MAX(d6, dist_[15]);
	dist_[7]  = MIN(d7, dist_[7]);
	dist_[16] = MAX(d7, dist_[16]);
	dist_[8]  = MIN(d8, dist_[8]);
	dist_[17] = MAX(d8, dist_[17]);
}

template <typename Scalar, int Dim>
void BoundingVolumeKDOP18<Scalar, Dim>::unionWith (const BoundingVolume* const bounding_volume)
{
	if(bounding_volume->getBVType() != getBVType())
		return;

	dist_[0]  = MIN(((BoundingVolumeKDOP18*)bounding_volume)->dist_[0], dist_[0]);
	dist_[9]  = MAX(((BoundingVolumeKDOP18*)bounding_volume)->dist_[9], dist_[9]);
	dist_[1]  = MIN(((BoundingVolumeKDOP18*)bounding_volume)->dist_[1], dist_[1]);
	dist_[10] = MAX(((BoundingVolumeKDOP18*)bounding_volume)->dist_[10], dist_[10]);
	dist_[2]  = MIN(((BoundingVolumeKDOP18*)bounding_volume)->dist_[2], dist_[2]);
	dist_[11] = MAX(((BoundingVolumeKDOP18*)bounding_volume)->dist_[11], dist_[11]);
	dist_[3]  = MIN(((BoundingVolumeKDOP18*)bounding_volume)->dist_[3], dist_[3]);
	dist_[12] = MAX(((BoundingVolumeKDOP18*)bounding_volume)->dist_[12], dist_[12]);
	dist_[4]  = MIN(((BoundingVolumeKDOP18*)bounding_volume)->dist_[4], dist_[4]);
	dist_[13] = MAX(((BoundingVolumeKDOP18*)bounding_volume)->dist_[13], dist_[13]);
	dist_[5]  = MIN(((BoundingVolumeKDOP18*)bounding_volume)->dist_[5], dist_[5]);
	dist_[14] = MAX(((BoundingVolumeKDOP18*)bounding_volume)->dist_[14], dist_[14]);
	dist_[6]  = MIN(((BoundingVolumeKDOP18*)bounding_volume)->dist_[6], dist_[6]);
	dist_[15] = MAX(((BoundingVolumeKDOP18*)bounding_volume)->dist_[15], dist_[15]);
	dist_[7]  = MIN(((BoundingVolumeKDOP18*)bounding_volume)->dist_[7], dist_[7]);
	dist_[16] = MAX(((BoundingVolumeKDOP18*)bounding_volume)->dist_[16], dist_[16]);
	dist_[8]  = MIN(((BoundingVolumeKDOP18*)bounding_volume)->dist_[8], dist_[8]);
	dist_[17] = MAX(((BoundingVolumeKDOP18*)bounding_volume)->dist_[17], dist_[17]);
}

template <typename Scalar, int Dim>
void BoundingVolumeKDOP18<Scalar, Dim>::obtainUnion(const BoundingVolume* const bounding_volume_lhs, const BoundingVolume* const bounding_volume_rhs)
{
	if(bounding_volume_lhs->getBVType() != getBVType())
		return;
	if(bounding_volume_rhs->getBVType() != getBVType())
		return;

	dist_[0]  = MIN(((BoundingVolumeKDOP18*)bounding_volume_lhs)->dist_[0], ((BoundingVolumeKDOP18*)bounding_volume_rhs)->dist_[0]);
	dist_[9]  = MAX(((BoundingVolumeKDOP18*)bounding_volume_lhs)->dist_[9], ((BoundingVolumeKDOP18*)bounding_volume_rhs)->dist_[9]);
	dist_[1]  = MIN(((BoundingVolumeKDOP18*)bounding_volume_lhs)->dist_[1], ((BoundingVolumeKDOP18*)bounding_volume_rhs)->dist_[1]);
	dist_[10] = MAX(((BoundingVolumeKDOP18*)bounding_volume_lhs)->dist_[10], ((BoundingVolumeKDOP18*)bounding_volume_rhs)->dist_[10]);
	dist_[2]  = MIN(((BoundingVolumeKDOP18*)bounding_volume_lhs)->dist_[2], ((BoundingVolumeKDOP18*)bounding_volume_rhs)->dist_[2]);
	dist_[11] = MAX(((BoundingVolumeKDOP18*)bounding_volume_lhs)->dist_[11], ((BoundingVolumeKDOP18*)bounding_volume_rhs)->dist_[11]);
	dist_[3]  = MIN(((BoundingVolumeKDOP18*)bounding_volume_lhs)->dist_[3], ((BoundingVolumeKDOP18*)bounding_volume_rhs)->dist_[3]);
	dist_[12] = MAX(((BoundingVolumeKDOP18*)bounding_volume_lhs)->dist_[12], ((BoundingVolumeKDOP18*)bounding_volume_rhs)->dist_[12]);
	dist_[4]  = MIN(((BoundingVolumeKDOP18*)bounding_volume_lhs)->dist_[4], ((BoundingVolumeKDOP18*)bounding_volume_rhs)->dist_[4]);
	dist_[13] = MAX(((BoundingVolumeKDOP18*)bounding_volume_lhs)->dist_[13], ((BoundingVolumeKDOP18*)bounding_volume_rhs)->dist_[13]);
	dist_[5]  = MIN(((BoundingVolumeKDOP18*)bounding_volume_lhs)->dist_[5], ((BoundingVolumeKDOP18*)bounding_volume_rhs)->dist_[5]);
	dist_[14] = MAX(((BoundingVolumeKDOP18*)bounding_volume_lhs)->dist_[14], ((BoundingVolumeKDOP18*)bounding_volume_rhs)->dist_[14]);
	dist_[6]  = MIN(((BoundingVolumeKDOP18*)bounding_volume_lhs)->dist_[6], ((BoundingVolumeKDOP18*)bounding_volume_rhs)->dist_[6]);
	dist_[15] = MAX(((BoundingVolumeKDOP18*)bounding_volume_lhs)->dist_[15], ((BoundingVolumeKDOP18*)bounding_volume_rhs)->dist_[15]);
	dist_[7]  = MIN(((BoundingVolumeKDOP18*)bounding_volume_lhs)->dist_[7], ((BoundingVolumeKDOP18*)bounding_volume_rhs)->dist_[7]);
	dist_[16] = MAX(((BoundingVolumeKDOP18*)bounding_volume_lhs)->dist_[16], ((BoundingVolumeKDOP18*)bounding_volume_rhs)->dist_[16]);
	dist_[8]  = MIN(((BoundingVolumeKDOP18*)bounding_volume_lhs)->dist_[8], ((BoundingVolumeKDOP18*)bounding_volume_rhs)->dist_[8]);
	dist_[17] = MAX(((BoundingVolumeKDOP18*)bounding_volume_lhs)->dist_[17], ((BoundingVolumeKDOP18*)bounding_volume_rhs)->dist_[17]);
}

//explicit instantitation
template class BoundingVolumeKDOP18<float, 3>;
template class BoundingVolumeKDOP18<double, 3>;

}