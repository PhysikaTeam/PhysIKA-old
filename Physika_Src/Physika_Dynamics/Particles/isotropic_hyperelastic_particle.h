/*
 * @file isotropic_hyperelastic_particle.h 
 * @Brief Solid particle with the isotropic hyperelastic constitutive model
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

#ifndef PHYSIKA_DYNAMICS_PARTICLES_ISOTROPIC_HYPERELASTIC_PARTICLE_H_
#define PHYSIKA_DYNAMICS_PARTICLES_ISOTROPIC_HYPERELASTIC_PARTICLE_H_

#include "Physika_Dynamics/Particles/solid_particle.h"

namespace Physika{

template <typename Scalar, int Dim> class IsotropicHyperelasticMaterial;

template <typename Scalar, int Dim>
class IsotropicHyperelasticParticle: public SolidParticle<Scalar,Dim>
{
public:
    IsotropicHyperelasticParticle();
    IsotropicHyperelasticParticle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol);
    IsotropicHyperelasticParticle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol, const SquareMatrix<Scalar,Dim> &deform_grad, const IsotropicHyperelasticMaterial<Scalar,Dim> &material);
    IsotropicHyperelasticParticle(const IsotropicHyperelasticParticle<Scalar,Dim> &particle);
    virtual ~IsotropicHyperelasticParticle();
    virtual IsotropicHyperelasticParticle<Scalar,Dim>* clone() const;
    IsotropicHyperelasticParticle<Scalar,Dim>& operator= (const IsotropicHyperelasticParticle<Scalar,Dim> &particle);
    //return pointer to the constitutive model, return NULL if not set
    const IsotropicHyperelasticMaterial<Scalar,Dim>* constitutiveModel() const;
    IsotropicHyperelasticMaterial<Scalar,Dim>* constitutiveModel();
    void setConstitutiveModel(const IsotropicHyperelasticMaterial<Scalar,Dim> &material);
protected:
    IsotropicHyperelasticMaterial<Scalar,Dim> *constitutive_model_;
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_PARTICLES_ISOTROPIC_HYPERELASTIC_PARTICLE_H_
