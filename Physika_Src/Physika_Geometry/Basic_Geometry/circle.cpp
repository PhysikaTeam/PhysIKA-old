/*
 * @file circle.cpp
 * @brief 2D counterpart of sphere. 
 * @author FeiZhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <cstdlib>
#include <iostream>
#include "Physika_Geometry/Basic_Geometry/circle.h"

namespace Physika{

template <typename Scalar>
Circle<Scalar>::Circle()
    :center_(Vector<Scalar,2>(0)),radius_(0)
{
}

template <typename Scalar>
Circle<Scalar>::Circle(const Vector<Scalar,2> &center, Scalar radius)
    :center_(center)
{
    setRadius(radius);
}

template <typename Scalar>
Circle<Scalar>::Circle(const Circle<Scalar> &circle)
    :center_(circle.center_),radius_(circle.radius_)
{
}

template <typename Scalar>
Circle<Scalar>::~Circle()
{
}

template <typename Scalar>
Circle<Scalar>& Circle<Scalar>::operator= (const Circle<Scalar> &circle)
{
    center_ = circle.center_;
    radius_ = circle.radius_;
    return *this;
}

template <typename Scalar>
Circle<Scalar>* Circle<Scalar>::clone() const
{
    return new Circle<Scalar>(*this);
}

template <typename Scalar>
void Circle<Scalar>::printInfo() const
{
//    std::cout<<"2D circle, center: "<<center_<<", radius: "<<radius_<<std::endl;
}

template <typename Scalar>
Vector<Scalar,2> Circle<Scalar>::center() const
{
    return center_;
}

template <typename Scalar>
void Circle<Scalar>::setCenter(const Vector<Scalar,2> &center)
{
    center_ = center;
}

template <typename Scalar>
Scalar Circle<Scalar>::radius() const
{
    return radius_;
}

template <typename Scalar>
void Circle<Scalar>::setRadius(Scalar radius)
{
    if(radius<0)
    {
        std::cerr<<"Warning: invalid radius provided, default value (0) is used instead!\n";
        radius_ = 0;
    }
    else
        radius_ = radius;
}

template <typename Scalar>
Scalar Circle<Scalar>::distance(const Vector<Scalar,2> &point) const
{
    Scalar signed_distance = signedDistance(point);
    return signed_distance>=0?signed_distance:-signed_distance;
}

template <typename Scalar>
Scalar Circle<Scalar>::signedDistance(const Vector<Scalar,2> &point) const
{
    return (point-center_).norm()-radius_;
}

template <typename Scalar>
bool Circle<Scalar>::inside(const Vector<Scalar,2> &point) const
{
    if(signedDistance(point)<=0)
        return true;
    else
        return false;
}

template <typename Scalar>
bool Circle<Scalar>::outside(const Vector<Scalar,2> &point) const
{
    return !inside(point);
}

template <typename Scalar>
Vector<Scalar,2> Circle<Scalar>::normal(const Vector<Scalar,2> &point) const
{
    Vector<Scalar,2> normal = point - center_;
    normal.normalize();
    return normal;
}

//explicit instantiation
template class Circle<float>;
template class Circle<double>;

}  //end of namespace Physika
