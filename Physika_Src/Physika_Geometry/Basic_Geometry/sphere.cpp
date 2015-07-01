/*
 * @file sphere.cpp
 * @brief 3D sphere class. 
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
    setRadius(radius);
}

template <typename Scalar>
Sphere<Scalar>::Sphere(const Sphere<Scalar> &sphere)
    :center_(sphere.center_),radius_(sphere.radius_)
{
}

template <typename Scalar>
Sphere<Scalar>::~Sphere()
{
}

template <typename Scalar>
Sphere<Scalar>& Sphere<Scalar>::operator= (const Sphere<Scalar> &sphere)
{
    center_ = sphere.center_;
    radius_ = sphere.radius_;
    return *this;
}

template <typename Scalar>
Sphere<Scalar>* Sphere<Scalar>::clone() const
{
    return new Sphere<Scalar>(*this);
}

template <typename Scalar>
void Sphere<Scalar>::printInfo() const
{
    std::cout<<"3D sphere, center: "<<center_<<", radius: "<<radius_<<std::endl;
}

template <typename Scalar>
Vector<Scalar,3> Sphere<Scalar>::center() const
{
    return center_;
}

template <typename Scalar>
void Sphere<Scalar>::setCenter(const Vector<Scalar,3> &center)
{
    center_ = center;
}

template <typename Scalar>
Scalar Sphere<Scalar>::radius() const
{
    return radius_;
}

template <typename Scalar>
void Sphere<Scalar>::setRadius(Scalar radius)
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

template <typename Scalar>
Vector<Scalar,3> Sphere<Scalar>::normal(const Vector<Scalar,3> &point) const
{
    Vector<Scalar,3> normal = point - center_;
    normal.normalize();
    return normal;
}

//explicit instantiation
template class Sphere<float>;
template class Sphere<double>;

}  //end of namespace Physika
