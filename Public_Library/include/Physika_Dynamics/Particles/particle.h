/*
 * @file particle.h 
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

#ifndef PHYSIKA_DYNAMICS_PARTICLES_PARTICLE_H_
#define PHYSIKA_DYNAMICS_PARTICLES_PARTICLE_H_

#include "Physika_Core/Vectors/vector.h"

namespace Physika{

template <typename Scalar, int Dim>
class Particle
{
public:
    Particle();
    Particle(const Scalar *pos, const Scalar *vel, Scalar mass, Scalar vol);
    Particle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol);
    Particle(const Particle<Scalar,Dim> &);
    ~Particle();
    void setPosition(const Scalar*);//set with a c/c++ array, assume array of proper size
    void setPosition(const Vector<Scalar,Dim> &);
    Vector<Scalar,Dim> position() const;
    void setVelocity(const Scalar*);
    void setVelocity(const Vector<Scalar,Dim> &);
    Vector<Scalar,Dim> velocity() const;
    void setMass(Scalar);
    Scalar mass() const;
    void setVolume(Scalar);
    Scalar volume() const;
protected:
    Scalar x_[Dim];
    Scalar v_[Dim];
    Scalar m_;
    Scalar vol_;
};

} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_PARTICLES_PARTICLE_H_
