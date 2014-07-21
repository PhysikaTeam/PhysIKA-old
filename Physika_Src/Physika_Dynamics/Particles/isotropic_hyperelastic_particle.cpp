/*
 * @file isotropic_hyperelastic_particle.cpp 
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

#include "Physika_Dynamics/Constitutive_Models/isotropic_hyperelastic_material.h"
#include "Physika_Dynamics/Particles/isotropic_hyperelastic_particle.h"

namespace Physika{

template <typename Scalar, int Dim>
IsotropicHyperelasticParticle<Scalar,Dim>::IsotropicHyperelasticParticle()
    :SolidParticle<Scalar,Dim>(),constitutive_model_(NULL)
{
}

template <typename Scalar, int Dim>
IsotropicHyperelasticParticle<Scalar,Dim>::IsotropicHyperelasticParticle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol)
    :SolidParticle<Scalar,Dim>(pos,vel,mass,vol),constitutive_model_(NULL)
{
}

template <typename Scalar, int Dim>
IsotropicHyperelasticParticle<Scalar,Dim>::IsotropicHyperelasticParticle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol, const SquareMatrix<Scalar,Dim> &deform_grad, const IsotropicHyperelasticMaterial<Scalar,Dim> &material)
    :SolidParticle<Scalar,Dim>(pos,vel,mass,vol,deform_grad),constitutive_model_(NULL)
{
    setConsitutiveModel(material);
}

template <typename Scalar, int Dim>
IsotropicHyperelasticParticle<Scalar,Dim>::IsotropicHyperelasticParticle(const IsotropicHyperelasticParticle<Scalar,Dim> &particle)
    :SolidParticle<Scalar,Dim>(particle),constitutive_model_(NULL)
{
    setConstitutiveModel(*(particle.material_));
}

template <typename Scalar, int Dim>
IsotropicHyperelasticParticle<Scalar,Dim>::~IsotropicHyperelasticParticle()
{
    if(constitutive_model_)
        delete constitutive_model_;
}

template <typename Scalar, int Dim>
IsotropicHyperelasticParticle<Scalar,Dim>& IsotropicHyperelasticParticle<Scalar,Dim>::operator= (const IsotropicHyperelasticParticle<Scalar,Dim> &particle)
{
    SolidParticle<Scalar,Dim>::operator= (particle);
    setConstitutiveModel(*(particle.material_));
    return *this;
}

template <typename Scalar, int Dim>
const IsotropicHyperelasticMaterial<Scalar,Dim>* IsotropicHyperelasticParticle<Scalar,Dim>::constitutiveModel() const
{
    return constitutive_model_;
}

template <typename Scalar, int Dim>
IsotropicHyperelasticMaterial<Scalar,Dim>* IsotropicHyperelasticParticle<Scalar,Dim>::constitutiveModel()
{
    return constitutive_model_;
}

template <typename Scalar, int Dim>
void IsotropicHyperelasticParticle<Scalar,Dim>::setConstitutiveModel(const IsotropicHyperelasticMaterial<Scalar,Dim> &material)
{
//TO DO
}

}  //end of namespace Physika
