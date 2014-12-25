/*
 * @file  implicit_collidable_object.h
 * @brief collidable object based on signed distance representation of object
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

#ifndef PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_IMPLICIT_COLLIDABLE_OBJECT_H_
#define PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_IMPLICIT_COLLIDABLE_OBJECT_H_

namespace Physika{

template <typename Scalar,int Dim>
class ImplicitCollidableObject: public CollidableObject<Scalar,Dim>
{
public:
protected:
};

}  //end of namespace Physika

#endif  //PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_IMPLICIT_COLLIDABLE_OBJECT_H_
