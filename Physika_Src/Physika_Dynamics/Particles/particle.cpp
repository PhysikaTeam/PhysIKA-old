/*
 * @file particle.cpp 
 * @Basic particle class. Particles used in solid&&fluid simulations inherit from this class.
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

#include "Physika_Dynamics/Particles/particle.h"

namespace Physika{

template <typename Scalar, int Dim>
Particle<Scalar,Dim>::Particle()
    :x_(0),v_(0),m_(0),vol_(0)
{
}

template <typename Scalar, int Dim>
Particle<Scalar,Dim>::Particle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol)
    :x_(pos),v_(vel),m_(mass),vol_(vol)
{
}

template <typename Scalar, int Dim>
Particle<Scalar,Dim>::Particle(const Particle<Scalar, Dim> &particle2)
{
    x_ = particle2.x_;
    v_ = particle2.v_;
    m_ = particle2.m_;
    vol_ = particle2.vol_;
}

template <typename Scalar, int Dim>
Particle<Scalar,Dim>::~Particle()
{
}

template <typename Scalar, int Dim>
Particle<Scalar,Dim>* Particle<Scalar,Dim>::clone() const
{
    return new Particle<Scalar,Dim>(*this);
}

template <typename Scalar, int Dim>
Particle<Scalar,Dim>& Particle<Scalar,Dim>::operator= (const Particle<Scalar,Dim> &particle2)
{
    x_ = particle2.x_;
    v_ = particle2.v_;
    m_ = particle2.m_;
    vol_ = particle2.vol_;
    return *this;
}

template <typename Scalar, int Dim>
void Particle<Scalar,Dim>::setPosition(const Vector<Scalar,Dim> &pos)
{
    x_ = pos;
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> Particle<Scalar,Dim>::position() const
{
    return x_;
}

template <typename Scalar, int Dim>
void Particle<Scalar,Dim>::setVelocity(const Vector<Scalar,Dim> &vel)
{
    v_ = vel;
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> Particle<Scalar,Dim>::velocity() const
{
    return v_;
}

template <typename Scalar, int Dim>
void Particle<Scalar,Dim>::setMass(Scalar mass)
{
    m_ = mass;
}

template <typename Scalar, int Dim>
Scalar Particle<Scalar,Dim>::mass() const
{
    return m_;
}

template <typename Scalar, int Dim>
void Particle<Scalar,Dim>::setVolume(Scalar vol)
{
    vol_ = vol;
}

template <typename Scalar, int Dim>
Scalar Particle<Scalar,Dim>::volume() const
{
    return vol_;
}

//explicit instantiation
template class Particle<float,2>;
template class Particle<double,2>;
template class Particle<float,3>;
template class Particle<double,3>;

} //end of namespace Physika
