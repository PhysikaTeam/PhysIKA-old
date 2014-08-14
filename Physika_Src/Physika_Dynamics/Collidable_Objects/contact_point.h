/*
 * @file  contact_point.h
 * @contact point of rigid body simulaion
 * @author Tianxiang Zhang
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_CONTACT_POINT_H_
#define PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_CONTACT_POINT_H_

#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Vectors/vector_2d.h"

namespace Physika{

/*
 * ContactPoint contains position and normal of contact points.
 * It's different from CollisionPair, which contains colliding elements (e.g. faces of meshes) and objects.
 * ContactPoint should be generated from CollisionPair. This process is called "contact sampling".
 * Notice that contact sampling is not a bijection.
 */
template <typename Scalar,int Dim>
class ContactPoint
{
public:
    ContactPoint();
    ContactPoint(unsigned int contact_index, unsigned int object_lhs_index, unsigned int object_rhs_index,
                const Vector<Scalar, Dim>& global_contact_position, const Vector<Scalar, Dim>& global_contact_normal_lhs);
    virtual ~ContactPoint();

    //get & set
    virtual void setProperty(unsigned int contact_index, unsigned int object_lhs_index, unsigned int object_rhs_index,
                             const Vector<Scalar, Dim>& global_contact_position, const Vector<Scalar, Dim>& global_contact_normal_lhs);
    unsigned int contactIndex() const;
    unsigned int objectLhsIndex() const;
    unsigned int objectRhsIndex() const;
    Vector<Scalar, Dim> globalContactPosition() const;
    Vector<Scalar, Dim> globalContactNormalLhs() const;
    Vector<Scalar, Dim> globalContactNormalRhs() const;

protected:
    unsigned int contact_index_;
    unsigned int object_lhs_index_;
    unsigned int object_rhs_index_;
    Vector<Scalar, Dim> global_contact_position_;
    Vector<Scalar, Dim> global_contact_normal_lhs_;//global contact normal of lhs object. global_contact_normal_rhs = -global_contact_normal_lhs
};

} //end of namespace Physika


#endif //PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_CONTACT_POINT_H_
