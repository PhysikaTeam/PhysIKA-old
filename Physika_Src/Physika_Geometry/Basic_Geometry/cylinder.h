/*
 * @file cylinder.h
 * @brief axis-aligned cylinder 
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

#ifndef PHYSIKA_GEOMETRY_BASIC_GEOMETRY_CYLINDER_H_
#define PHYSIKA_GEOMETRY_BASIC_GEOMETRY_CYLINDER_H_

#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Basic_Geometry/basic_geometry.h"

namespace Physika{

template <typename Scalar>
class Cylinder: public BasicGeometry<Scalar,3>
{
public:
    Cylinder();
    Cylinder(const Vector<Scalar,3> &center, Scalar radius, Scalar length, unsigned int direction);
    Cylinder(const Cylinder<Scalar> &cylinder);
    ~Cylinder();
    Cylinder<Scalar>& operator= (const Cylinder<Scalar> &cylinder);
    Cylinder<Scalar>* clone() const;
    virtual void printInfo() const;
    Vector<Scalar,3> center() const;
    void setCenter(const Vector<Scalar,3> &center);
    Scalar radius() const;
    void setRadius(Scalar radius);
    Scalar length() const;
    void setLength(Scalar length);
    virtual Scalar distance(const Vector<Scalar,3> &point) const;
    virtual Scalar signedDistance(const Vector<Scalar,3> &point) const;
    bool inside(const Vector<Scalar,3> &point) const;
    bool outside(const Vector<Scalar,3> &point) const;
    virtual Vector<Scalar,3> normal(const Vector<Scalar,3> &point) const;
protected:
    Vector<Scalar,3> center_;
    Scalar radius_;
    Scalar length_;
    unsigned int direction_; //0: x; 1: y; 2: z
};

}  //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_BASIC_GEOMETRY_CYLINDER_H_
