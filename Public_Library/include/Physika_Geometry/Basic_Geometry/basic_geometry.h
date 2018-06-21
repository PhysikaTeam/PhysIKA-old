/*
 * @file basic_geometry.h
 * @brief base class of all basic geometry like plane, sphere, etc. 
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

#ifndef PHYSIKA_GEOMETRY_BASIC_GEOMETRY_BASIC_GEOMETRY_H_
#define PHYSIKA_GEOMETRY_BASIC_GEOMETRY_BASIC_GEOMETRY_H_

namespace Physika{

template <typename Scalar, int Dim> class Vector;

/*
 * Note on signed distance: positive outside, negative inside
 *
 */

template <typename Scalar, int Dim>
class BasicGeometry
{
public:
    BasicGeometry();
    BasicGeometry(const BasicGeometry<Scalar,Dim> &geometry);
    virtual ~BasicGeometry();
    BasicGeometry<Scalar,Dim>& operator= (const BasicGeometry<Scalar,Dim> &geometry);
    virtual BasicGeometry<Scalar,Dim>* clone() const = 0;
    virtual void printInfo() const = 0;
    virtual Vector<Scalar,Dim> normal(const Vector<Scalar,Dim> &point) const = 0;
    virtual Scalar distance(const Vector<Scalar,Dim> &point) const = 0; //distance to the surface of geometry
    virtual Scalar signedDistance(const Vector<Scalar,Dim> &point) const = 0;  //signed distance to the surface of geometry
protected:
};

}  //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_BASIC_GEOMETRY_BASIC_GEOMETRY_H_
