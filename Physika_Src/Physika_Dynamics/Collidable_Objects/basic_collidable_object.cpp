/*
 * @file basic_collidable_object.cpp 
 * @brief basic geometry based collidable object
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <limits>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Geometry/Basic_Geometry/basic_geometry.h"
#include "Physika_Dynamics/Collidable_Objects/basic_collidable_object.h"

namespace Physika{

template <typename Scalar, int Dim>
BasicCollidableObject<Scalar,Dim>::BasicCollidableObject()
    :CollidableObject<Scalar,Dim>(),shape_(NULL)
{
}

template <typename Scalar, int Dim>
BasicCollidableObject<Scalar,Dim>::BasicCollidableObject(const BasicGeometry<Scalar,Dim> &geometry)
    :CollidableObject<Scalar,Dim>()
{
    shape_ = geometry.clone();
}

template <typename Scalar, int Dim>
BasicCollidableObject<Scalar,Dim>::BasicCollidableObject(Scalar mu, bool sticky, const BasicGeometry<Scalar,Dim> &geometry)
    :CollidableObject<Scalar,Dim>(mu,sticky)
{
    shape_ = geometry.clone();
}

template <typename Scalar, int Dim>
BasicCollidableObject<Scalar,Dim>::BasicCollidableObject(const BasicCollidableObject<Scalar,Dim> &object)
    :CollidableObject<Scalar,Dim>(object)
{
    shape_ = object.shape_->clone();
}

template <typename Scalar, int Dim>
BasicCollidableObject<Scalar,Dim>::~BasicCollidableObject()
{
    if(shape_)
        delete shape_;
}

template <typename Scalar, int Dim>
BasicCollidableObject<Scalar,Dim>& BasicCollidableObject<Scalar,Dim>::operator= (const BasicCollidableObject<Scalar,Dim> &object)
{
    CollidableObject<Scalar,Dim>::operator= (object);
    shape_ = object.shape_->clone();
    return *this;
}

template <typename Scalar, int Dim>
BasicCollidableObject<Scalar,Dim>* BasicCollidableObject<Scalar,Dim>::clone() const
{
    return new BasicCollidableObject<Scalar,Dim>(*this);
}

template <typename Scalar, int Dim>
bool BasicCollidableObject<Scalar,Dim>::collide(const Vector<Scalar,Dim> &point, const Vector<Scalar,Dim> &velocity, Vector<Scalar,Dim> &velocity_impulse) const
{
    PHYSIKA_ASSERT(shape_);
    bool collide = false;
    Scalar signed_distance = shape_->signedDistance(point);
    Vector<Scalar,Dim> normal = shape_->normal(point);
    Vector<Scalar,Dim> vel_delta = velocity - this->velocity_;
    Scalar projection = vel_delta.dot(normal); 
    if(projection < 0 && signed_distance < this->collide_threshold_)
    {
        collide = true;
        if(this->sticky_)
            velocity_impulse = -vel_delta;
        else
        {
            Vector<Scalar,Dim> vel_delta_t =vel_delta - projection*normal;
            Scalar tangent_length = vel_delta_t.norm();
            if(-projection*(this->mu_) < tangent_length)
            {
                velocity_impulse = -projection*normal; //normal direction
                if(tangent_length > std::numeric_limits<Scalar>::epsilon())
                    velocity_impulse += (vel_delta_t/tangent_length)*projection*(this->mu_); //tangent direction
            }
            else
                velocity_impulse = -vel_delta;
        }
    }
    return collide;
}

template <typename Scalar, int Dim>
Scalar BasicCollidableObject<Scalar,Dim>::distance(const Vector<Scalar,Dim> &point) const
{
    PHYSIKA_ASSERT(shape_);
    return shape_->distance(point);
}

template <typename Scalar, int Dim>
Scalar BasicCollidableObject<Scalar,Dim>::signedDistance(const Vector<Scalar,Dim> &point) const
{
    PHYSIKA_ASSERT(shape_);
    return shape_->signedDistance(point);
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> BasicCollidableObject<Scalar,Dim>::normal(const Vector<Scalar,Dim> &point) const
{
    PHYSIKA_ASSERT(shape_);
    return shape_->normal(point);
}

template <typename Scalar, int Dim>
const BasicGeometry<Scalar,Dim>& BasicCollidableObject<Scalar,Dim>::shape() const
{
    PHYSIKA_ASSERT(shape_);
    return *shape_;
}

template <typename Scalar, int Dim>
BasicGeometry<Scalar,Dim>& BasicCollidableObject<Scalar,Dim>::shape()
{
    PHYSIKA_ASSERT(shape_);
    return *shape_;
}

template <typename Scalar, int Dim>
void BasicCollidableObject<Scalar,Dim>::setShape(const BasicGeometry<Scalar,Dim> &geometry)
{
    if(shape_)
        delete shape_;
    shape_ = geometry.clone();
}

//explicit instantiations
template class BasicCollidableObject<float,2>;
template class BasicCollidableObject<float,3>;
template class BasicCollidableObject<double,2>;
template class BasicCollidableObject<double,3>;

}  //end of namespace Physika
