/*
 * @file  contact_point.cpp
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

#include <limits>
#include "Physika_Dynamics/Collidable_Objects/contact_point.h"

namespace Physika{

template <typename Scalar,int Dim>
ContactPoint<Scalar, Dim>::ContactPoint():
    contact_index_(0),
    object_lhs_index_(0),
    object_rhs_index_(0),
    global_contact_position_(0),
    global_contact_normal_lhs_(0)
{

}

template <typename Scalar,int Dim>
ContactPoint<Scalar, Dim>::ContactPoint(unsigned int contact_index, unsigned int object_lhs_index, unsigned int object_rhs_index,
    const Vector<Scalar, Dim>& global_contact_position, const Vector<Scalar, Dim>& global_contact_normal_lhs):
    contact_index_(contact_index),
    object_lhs_index_(object_lhs_index),
    object_rhs_index_(object_rhs_index),
    global_contact_position_(global_contact_position),
    global_contact_normal_lhs_(global_contact_normal_lhs)
{

}

template <typename Scalar,int Dim>
ContactPoint<Scalar, Dim>::~ContactPoint()
{

}

template <typename Scalar,int Dim>
void ContactPoint<Scalar, Dim>::setProperty(unsigned int contact_index, unsigned int object_lhs_index, unsigned int object_rhs_index,
    const Vector<Scalar, Dim>& global_contact_position, const Vector<Scalar, Dim>& global_contact_normal_lhs)
{
    contact_index_ = contact_index;
    object_lhs_index_ = object_lhs_index;
    object_rhs_index_ = object_rhs_index;
    global_contact_position_ = global_contact_position;
    global_contact_normal_lhs_ = global_contact_normal_lhs;
}

template <typename Scalar,int Dim>
unsigned int ContactPoint<Scalar, Dim>::contactIndex() const
{
    return contact_index_;
}

template <typename Scalar,int Dim>
unsigned int ContactPoint<Scalar, Dim>::objectLhsIndex() const
{
    return object_lhs_index_;
}

template <typename Scalar,int Dim>
unsigned int ContactPoint<Scalar, Dim>::objectRhsIndex() const
{
    return object_rhs_index_;
}

template <typename Scalar,int Dim>
Vector<Scalar, Dim> ContactPoint<Scalar, Dim>::globalContactPosition() const
{
    return global_contact_position_;
}

template <typename Scalar,int Dim>
Vector<Scalar, Dim> ContactPoint<Scalar, Dim>::globalContactNormalLhs() const
{
    return global_contact_normal_lhs_;
}

template <typename Scalar,int Dim>
Vector<Scalar, Dim> ContactPoint<Scalar, Dim>::globalContactNormalRhs() const
{
    return -global_contact_normal_lhs_;
}


template class ContactPoint<float, 2>;
template class ContactPoint<double, 2>;
template class ContactPoint<float, 3>;
template class ContactPoint<double, 3>;

}
