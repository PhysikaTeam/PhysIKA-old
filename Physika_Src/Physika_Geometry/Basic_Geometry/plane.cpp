/*
 * @file plane.cpp
 * @brief 3D plane class. 
 * @author FeiZhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <iostream>
#include "Physika_Geometry/Basic_Geometry/plane.h"

namespace Physika{

template <typename Scalar>
Plane<Scalar>::Plane()
{
}

template <typename Scalar>
Plane<Scalar>::Plane(const Vector<Scalar,3> &normal, const Vector<Scalar,3> &point_on_plane)
    :normal_(normal),pos_(point_on_plane)
{
    normal_.normalize();
}

template <typename Scalar>
Plane<Scalar>::Plane(const Vector<Scalar,3> &x1, const Vector<Scalar,3> &x2, const Vector<Scalar,3> &x3)
{
    Vector<Scalar,3> vec1 = x2 - x1, vec2 = x3 - x1;
    normal_ = vec1.cross(vec2);
    normal_.normalize();
    pos_ = x1;
}

template <typename Scalar>
Plane<Scalar>::~Plane()
{
}

template <typename Scalar>
void Plane<Scalar>::printInfo() const
{
    std::cout<<"3D plane, normal: "<<normal_<<std::endl;
}

template <typename Scalar>
const Vector<Scalar,3>& Plane<Scalar>::normal() const
{
    return normal_;
}

template <typename Scalar>
Scalar Plane<Scalar>::distance(const Vector<Scalar,3> &point) const
{
    Scalar signed_dist = signedDistance(point);
    Scalar dist = signed_dist>=0?signed_dist:-signed_dist;
    return dist;
}

template <typename Scalar>
Scalar Plane<Scalar>::signedDistance(const Vector<Scalar,3> &point) const
{
    Vector<Scalar,3> plane_to_point = point - pos_;
    Scalar dist = plane_to_point.dot(normal_);
    return dist;
}

//explicit instantitation
template class Plane<float>;
template class Plane<double>;

}  //end of namespace Physika














