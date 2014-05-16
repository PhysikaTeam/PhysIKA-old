/*
 * @file plane.h
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

#ifndef PHYSIKA_GEOMETRY_BASIC_GEOMETRY_PLANE_H_
#define PHYSIKA_GEOMETRY_BASIC_GEOMETRY_PLANE_H_

#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Basic_Geometry/basic_geometry.h"

namespace Physika{

template <typename Scalar>
class Plane: public BasicGeometry
{
public:
    Plane();
    Plane(const Vector<Scalar,3> &normal, const Vector<Scalar,3> &point_on_plane);
    //specify plane with 3 points, the order of the points determines the direction of the plane
    Plane(const Vector<Scalar,3> &x1, const Vector<Scalar,3> &x2, const Vector<Scalar,3> &x3);
    ~Plane();
    void printInfo() const;
    const Vector<Scalar,3>& normal() const;
    Scalar distance(const Vector<Scalar,3> &point) const;  //distance of given point to the plane
    Scalar signedDistance(const Vector<Scalar,3> &point) const; //signed distance of given point to the plane (consider plane direction)
protected:
    Vector<Scalar,3> normal_;  //normal of the plane
    Vector<Scalar,3> pos_;  //one point on the plane
};

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_BASIC_GEOMETRY_PLANE_H_
