/*
 * @file collidable_object.cpp 
 * @brief base class of all collidable objects. A collidable object is the kinematic object
 *            in scene that is used for simple contact effects.
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

#include <iostream>
#include "Physika_Dynamics/Collidable_Objects/collidable_object.h"

namespace Physika{

template <typename Scalar, int Dim>
CollidableObject<Scalar,Dim>::CollidableObject()
    :mu_(0), sticky_(false),collide_threshold_(0),velocity_(0)
{
}

template <typename Scalar, int Dim>
CollidableObject<Scalar,Dim>::CollidableObject(Scalar mu, bool sticky)
    :mu_(mu),sticky_(sticky),collide_threshold_(0),velocity_(0)
{
}

template <typename Scalar, int Dim>
CollidableObject<Scalar,Dim>::CollidableObject(const CollidableObject<Scalar,Dim> &object)
    :mu_(object.mu_),sticky_(object.sticky_),
     collide_threshold_(object.collide_threshold_),velocity_(object.velocity_)
{
}

template <typename Scalar, int Dim>
CollidableObject<Scalar,Dim>::~CollidableObject()
{
}

template <typename Scalar, int Dim>
CollidableObject<Scalar,Dim>& CollidableObject<Scalar,Dim>::operator= (const CollidableObject<Scalar,Dim> &object)
{
    mu_ = object.mu_;
    sticky_ = object.sticky_;
    collide_threshold_ = object.collide_threshold_;
    velocity_ = object.velocity_;
    return *this;
}

template <typename Scalar, int Dim>
Scalar CollidableObject<Scalar,Dim>::mu() const
{
    return mu_;
}

template <typename Scalar, int Dim>
void CollidableObject<Scalar,Dim>::setMu(Scalar mu)
{
    if(mu < 0)
    {
        std::cerr<<"Warning: invalid mu provided, default value (0) is used instead!\n";
        mu_ = 0;
    }
    else
        mu_ = mu;
}

template <typename Scalar, int Dim>
bool CollidableObject<Scalar,Dim>::isSticky() const
{
    return sticky_;
}

template <typename Scalar, int Dim>
void CollidableObject<Scalar,Dim>::setSticky(bool sticky)
{
    sticky_ = sticky;
}

template <typename Scalar, int Dim>
Scalar CollidableObject<Scalar,Dim>::collideThreshold() const
{
    return collide_threshold_;
}

template <typename Scalar, int Dim>
void CollidableObject<Scalar,Dim>::setCollideThreshold(Scalar threshold)
{
    if(threshold < 0)
    {
        std::cerr<<"Invalid threshold provided, default value (0) is used instead!\n";
        collide_threshold_ = 0;
    }
    else
        collide_threshold_ = threshold;
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> CollidableObject<Scalar,Dim>::velocity() const
{
    return velocity_;
}

template <typename Scalar, int Dim>
void CollidableObject<Scalar,Dim>::setVelocity(const Vector<Scalar,Dim> &velocity)
{
    velocity_ = velocity;
}

//explicit instantiations
template class CollidableObject<float,2>;
template class CollidableObject<float,3>;
template class CollidableObject<double,2>;
template class CollidableObject<double,3>;

}  //end of namespace Physika
