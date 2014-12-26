/*
 * @file basic_collidable_object.h 
 * @brief basic geometry based collidable object
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_BASIC_COLLIDABLE_OBJECT_H_
#define PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_BASIC_COLLIDABLE_OBJECT_H_

#include "Physika_Dynamics/Collidable_Objects/collidable_object.h"

namespace Physika{

template <typename Scalar, int Dim> class BasicGeometry;

template <typename Scalar, int Dim>
class BasicCollidableObject: public CollidableObject<Scalar,Dim>
{
public:
    explicit BasicCollidableObject(const BasicGeometry<Scalar,Dim> &geometry);
    BasicCollidableObject(Scalar mu, bool sticky, const BasicGeometry<Scalar,Dim> &geometry);
    BasicCollidableObject(const BasicCollidableObject<Scalar,Dim> &object);
    ~BasicCollidableObject();
    BasicCollidableObject<Scalar,Dim>& operator= (const BasicCollidableObject<Scalar,Dim> &object);
    BasicCollidableObject<Scalar,Dim>* clone() const;
    bool collide(const Vector<Scalar,Dim> &point, const Vector<Scalar,Dim> &velocity, Vector<Scalar,Dim> &velocity_impulse) const;
    const BasicGeometry<Scalar,Dim>& shape() const;
    BasicGeometry<Scalar,Dim>& shape();
    void setShape(const BasicGeometry<Scalar,Dim> &geometry);
protected:
    BasicCollidableObject(); //prohibit construction of collidable object with no geometry
protected:
    BasicGeometry<Scalar,Dim> *shape_;
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_BASIC_COLLIDABLE_OBJECT_H_
