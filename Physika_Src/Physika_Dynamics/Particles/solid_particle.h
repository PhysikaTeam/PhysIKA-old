/*
 * @file solid_particle.h 
 * @Brief the particle used to represent solid, carry deformation gradient information
 *        and constitutive model
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

#ifndef PHYSIKA_DYNAMICS_PARTICLES_SOLID_PARTICLE_H_
#define PHYSIKA_DYNAMICS_PARTICLES_SOLID_PARTICLE_H_

#include "Physika_Dynamics/Particles/particle.h"
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"

namespace Physika{

template <typename Scalar, int Dim> class ConstitutiveModel;
template <typename Scalar, int Dim> class Vector;

template <typename Scalar, int Dim>
class SolidParticle: public Particle<Scalar,Dim>
{
public:
    SolidParticle();
    SolidParticle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol); //F set to identity, constitutive model unset
    SolidParticle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol, const SquareMatrix<Scalar,Dim> &deform_grad);  //constitutive model unset
    SolidParticle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol, const SquareMatrix<Scalar,Dim> &deform_grad, const ConstitutiveModel<Scalar,Dim> &material);
    SolidParticle(const SolidParticle<Scalar,Dim> &particle);
    virtual ~SolidParticle(); 
    virtual SolidParticle<Scalar,Dim>* clone() const;
    SolidParticle<Scalar,Dim>& operator= (const SolidParticle<Scalar,Dim> &particle);
    SquareMatrix<Scalar,Dim> deformationGradient() const;
    const ConstitutiveModel<Scalar,Dim>& constitutiveModel() const;
    ConstitutiveModel<Scalar,Dim>& constitutiveModel();
    Scalar energy() const;
    SquareMatrix<Scalar,Dim> firstPiolaKirchhoffStress() const;
    SquareMatrix<Scalar,Dim> secondPiolaKirchhoffStress() const;
    SquareMatrix<Scalar,Dim> cauchyStress() const;
    void setDeformationGradient(const SquareMatrix<Scalar,Dim> &F);
    void setConstitutiveModel(const ConstitutiveModel<Scalar,Dim> &material);
protected:
    SquareMatrix<Scalar,Dim> F_;
    ConstitutiveModel<Scalar,Dim> *constitutive_model_;
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_PARTICLES_SOLID_PARTICLE_H_
