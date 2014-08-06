/*
 * @file particle.h 
 * @Basic particle class. Particles used in solid&&fluid simulations inherit from this class.
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

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

/*
 * Particle class for various simulations. 
 *
 */

template <typename Scalar, int Dim>
class Particle
{
public:
    Particle();
    Particle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol);  //weight and weight gradient unset
    Particle(const Particle<Scalar,Dim> &particle);
    virtual ~Particle();
    virtual Particle<Scalar,Dim>* clone() const;
    Particle<Scalar,Dim>& operator= (const Particle<Scalar,Dim> &particle);
    void setPosition(const Vector<Scalar,Dim> &);
    Vector<Scalar,Dim> position() const;
    void setVelocity(const Vector<Scalar,Dim> &);
    Vector<Scalar,Dim> velocity() const;
    void setMass(Scalar);
    Scalar mass() const;
    void setVolume(Scalar);
    Scalar volume() const;

protected:
    Vector<Scalar,Dim> x_;
    Vector<Scalar,Dim> v_;
    Scalar m_;
    Scalar vol_;
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_PARTICLES_PARTICLE_H_
