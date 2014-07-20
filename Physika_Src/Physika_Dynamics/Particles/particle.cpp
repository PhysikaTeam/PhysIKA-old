/*
 * @file particle.cpp 
 * @Basic particle class. Particles for fluid && solid simulation inherit from this class.
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

#include <iostream>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Dynamics/Particles/particle.h"

namespace Physika{

template <typename Scalar, int Dim>
Particle<Scalar,Dim>::Particle()
{
    for(int i = 0; i < Dim; ++i)
    {
        x_[i] = 0;
        v_[i] = 0;
    }
    m_ = 0;
    vol_ = 0;
}

template <typename Scalar, int Dim>
Particle<Scalar,Dim>::Particle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol)
{
    setPosition(pos);
    setVelocity(vel);
    m_ = mass;
    vol_ = vol;
}

template <typename Scalar, int Dim>
Particle<Scalar,Dim>::Particle(const Particle<Scalar, Dim> &particle2)
{
    setPosition(particle2.position());
    setVelocity(particle2.velocity());
    setMass(particle2.mass());
    setVolume(particle2.volume());
}

template <typename Scalar, int Dim>
Particle<Scalar,Dim>::~Particle()
{
}

template <typename Scalar, int Dim>
Particle<Scalar,Dim>& Particle<Scalar,Dim>::operator= (const Particle<Scalar,Dim> &particle2)
{
    setPosition(particle2.position());
    setVelocity(particle2.velocity());
    setMass(particle2.mass());
    setVolume(particle2.volume());
    return *this;
}

template <typename Scalar, int Dim>
void Particle<Scalar,Dim>::setPosition(const Vector<Scalar,Dim> &pos)
{
    for(int i = 0; i < Dim; ++i)
        x_[i] = pos[i];
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> Particle<Scalar,Dim>::position() const
{
    Vector<Scalar,Dim> pos;
    for(int i = 0; i < Dim; ++i)
        pos[i] = x_[i];
    return pos;
}

template <typename Scalar, int Dim>
void Particle<Scalar,Dim>::setVelocity(const Vector<Scalar,Dim> &vel)
{
    for(int i = 0; i < Dim; ++i)
	v_[i] = vel[i];
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> Particle<Scalar,Dim>::velocity() const
{
    Vector<Scalar,Dim> vel;
    for(int i = 0; i < Dim; ++i)
        vel[i] = v_[i];
    return vel;
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
