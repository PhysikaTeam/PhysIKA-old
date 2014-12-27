/*
 * @file cylinder.cpp
 * @brief axis-aligned cylinder 
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
#include "Physika_Geometry/Basic_Geometry/cylinder.h"

namespace Physika{

template <typename Scalar>
Cylinder<Scalar>::Cylinder()
    :center_(0),radius_(0),length_(0),direction_(0)
{
}

template <typename Scalar>
Cylinder<Scalar>::Cylinder(const Vector<Scalar,3> &center, Scalar radius, Scalar length, unsigned int direction)
    :center_(center),radius_(radius),length_(length),direction_(direction)
{
}

template <typename Scalar>
Cylinder<Scalar>::Cylinder(const Cylinder<Scalar> &cylinder)
    :center_(cylinder.center_),radius_(cylinder.radius_),length_(cylinder.length_),direction_(cylinder.direction_)
{
}

template <typename Scalar>
Cylinder<Scalar>::~Cylinder()
{
}

template <typename Scalar>
Cylinder<Scalar>& Cylinder<Scalar>::operator= (const Cylinder<Scalar> &cylinder)
{
    center_= cylinder.center_;
    radius_ = cylinder.radius_;
    length_ = cylinder.length_;
    direction_ = cylinder.direction_;
    return *this;
}

template <typename Scalar>
Cylinder<Scalar>* Cylinder<Scalar>::clone() const
{
    return new Cylinder<Scalar>(*this);
}

template <typename Scalar>
void Cylinder<Scalar>::printInfo() const
{
    std::cout<<"Temp cylinder class, will be deleted soon!\n";
}

template <typename Scalar>
Vector<Scalar,3> Cylinder<Scalar>::center() const
{
    return center_;
}

template <typename Scalar>
void Cylinder<Scalar>::setCenter(const Vector<Scalar,3> &center)
{
    center_ = center;
}

template <typename Scalar>
Scalar Cylinder<Scalar>::radius() const
{
    return radius_;
}

template <typename Scalar>
void Cylinder<Scalar>::setRadius(Scalar radius)
{
    radius_ = radius;
}

template <typename Scalar>
Scalar Cylinder<Scalar>::length() const
{
    return length_;
}

template <typename Scalar>
void Cylinder<Scalar>::setLength(Scalar length)
{
    length_ = length;
}

template <typename Scalar>
Scalar Cylinder<Scalar>::distance(const Vector<Scalar,3> &point) const
{
    Scalar signed_distance = signedDistance(point);
    return signed_distance >=0 ? signed_distance : -signed_distance;
}

template <typename Scalar> 
Scalar Cylinder<Scalar>::signedDistance(const Vector<Scalar,3> &point) const
{
    Vector<Scalar,3> new_center = center_;
    new_center[direction_] = point[direction_];
    return (point-new_center).norm()-radius_;
}

template <typename Scalar> 
bool Cylinder<Scalar>::inside(const Vector<Scalar,3> &point) const
{
    if(signedDistance(point)<=0)
        return true;
    else
        return false;
}

template <typename Scalar> 
bool Cylinder<Scalar>::outside(const Vector<Scalar,3> &point) const
{
    return !inside(point);
}

template <typename Scalar> 
Vector<Scalar,3> Cylinder<Scalar>::normal(const Vector<Scalar,3> &point) const
{
    Vector<Scalar,3> new_center = center_;
    new_center[direction_] = point[direction_];
    Vector<Scalar,3> normal = point - new_center;
    normal.normalize();
    return normal;
}

//explicit instantiations
template class Cylinder<float>;
template class Cylinder<double>;

}  //end of namespace Physika
