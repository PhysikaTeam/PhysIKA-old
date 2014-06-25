/*
 * @file sphere.cpp
 * @brief 3D sphere class. 
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

#include <cstdlib>
#include <iostream>
#include "Physika_Geometry/Basic_Geometry/sphere.h"

namespace Physika{

template <typename Scalar>
Sphere<Scalar>::Sphere()
    :center_(Vector<Scalar,3>(0)),radius_(0)
{
}

template <typename Scalar>
Sphere<Scalar>::Sphere(const Vector<Scalar,3> &center, Scalar radius)
    :center_(center)
{
    if(radius<0)
    {
        std::cerr<<"Radius of a sphere must be equal or greater than zero!\n";
        std::exit(EXIT_FAILURE);
    }
    radius_ = radius;
}

template <typename Scalar>
Sphere<Scalar>::~Sphere()
{
}

template <typename Scalar>
void Sphere<Scalar>::printInfo() const
{
    std::cout<<"3D sphere, center: "<<center_<<", radius: "<<radius_<<std::endl;
}

template <typename Scalar>
const Vector<Scalar,3>& Sphere<Scalar>::center() const
{
    return center_;
}

template <typename Scalar>
Scalar Sphere<Scalar>::radius() const
{
    return radius_;
}

template <typename Scalar>
Scalar Sphere<Scalar>::distance(const Vector<Scalar,3> &point) const
{
    Scalar signed_distance = signedDistance(point);
    return signed_distance>=0?signed_distance:-signed_distance;
}

template <typename Scalar>
Scalar Sphere<Scalar>::signedDistance(const Vector<Scalar,3> &point) const
{
    return (point-center_).norm()-radius_;
}

template <typename Scalar>
bool Sphere<Scalar>::inside(const Vector<Scalar,3> &point) const
{
    if(signedDistance(point)<=0)
        return true;
    else
        return false;
}

template <typename Scalar>
bool Sphere<Scalar>::outside(const Vector<Scalar,3> &point) const
{
    return !inside(point);
}

}  //end of namespace Physika


















