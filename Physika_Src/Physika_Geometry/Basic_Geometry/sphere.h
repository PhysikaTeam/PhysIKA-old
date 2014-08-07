/*
 * @file sphere.h
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

#ifndef PHYSIKA_GEOMETRY_BASIC_GEOMETRY_SPHERE_H_
#define PHYSIKA_GEOMETRY_BASIC_GEOMETRY_SPHERE_H_

#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Basic_Geometry/basic_geometry.h"

namespace Physika{

template <typename Scalar>
class Sphere: public BasicGeometry
{
public:
    Sphere();
    Sphere(const Vector<Scalar,3> &center, Scalar radius);
    ~Sphere();
    void printInfo() const;
    Vector<Scalar,3> center() const;
    Scalar radius() const;
    Scalar distance(const Vector<Scalar,3> &point) const;  //distance of a point to sphere surface
    Scalar signedDistance(const Vector<Scalar,3> &point) const;  //signed distance of a point to a sphere surface
    bool inside(const Vector<Scalar,3> &point) const;  //d <= radius
    bool outside(const Vector<Scalar,3> &point) const; //d > radius
protected:
    Vector<Scalar,3> center_;
    Scalar radius_;
};

}  //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_BASIC_GEOMETRY_SPHERE_H_
