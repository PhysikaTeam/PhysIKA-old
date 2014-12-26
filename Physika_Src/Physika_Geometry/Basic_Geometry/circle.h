/*
 * @file circle.h
 * @brief 2D counterpart of sphere 
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

#ifndef PHYSIKA_GEOMETRY_BASIC_GEOMETRY_CIRCLE_H_
#define PHYSIKA_GEOMETRY_BASIC_GEOMETRY_CIRCLE_H_

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Geometry/Basic_Geometry/basic_geometry.h"

namespace Physika{

template <typename Scalar>
class Circle: public BasicGeometry<Scalar,2>
{
public:
    Circle();
    Circle(const Vector<Scalar,2> &center, Scalar radius);
    Circle(const Circle<Scalar> &circle);
    ~Circle();
    Circle<Scalar>& operator= (const Circle<Scalar> &circle);
    Circle<Scalar>* clone() const;
    virtual void printInfo() const;
    Vector<Scalar,2> center() const;
    void setCenter(const Vector<Scalar,2> &center);
    Scalar radius() const;
    void setRadius(Scalar radius);
    virtual Scalar distance(const Vector<Scalar,2> &point) const;  //distance of a point to circle surface
    virtual Scalar signedDistance(const Vector<Scalar,2> &point) const;  //signed distance of a point to a circle surface
    bool inside(const Vector<Scalar,2> &point) const;  //d <= radius
    bool outside(const Vector<Scalar,2> &point) const; //d > radius
    virtual Vector<Scalar,2> normal(const Vector<Scalar,2> &point) const; 
protected:
    Vector<Scalar,2> center_;
    Scalar radius_;
};

}  //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_BASIC_GEOMETRY_CIRCLE_H_
