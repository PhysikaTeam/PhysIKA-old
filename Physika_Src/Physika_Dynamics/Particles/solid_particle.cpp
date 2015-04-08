/*
 * @file solid_particle.cpp 
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

#include <cstdlib>
#include <iostream>
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/Constitutive_Models/constitutive_model.h"

namespace Physika{

template <typename Scalar, int Dim>
SolidParticle<Scalar,Dim>::SolidParticle()
    :Particle<Scalar,Dim>(),constitutive_model_(NULL)
{
    F_ = SquareMatrix<Scalar,Dim>::identityMatrix();
}

template <typename Scalar, int Dim>
SolidParticle<Scalar,Dim>::SolidParticle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol)
    :Particle<Scalar,Dim>(pos,vel,mass,vol),constitutive_model_(NULL)
{
    F_ = SquareMatrix<Scalar,Dim>::identityMatrix();
}

template <typename Scalar, int Dim>
SolidParticle<Scalar,Dim>::SolidParticle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol, const SquareMatrix<Scalar,Dim> &deform_grad)
    :Particle<Scalar,Dim>(pos,vel,mass,vol),F_(deform_grad),constitutive_model_(NULL)
{
}

template <typename Scalar, int Dim>
SolidParticle<Scalar,Dim>::SolidParticle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol, const SquareMatrix<Scalar,Dim> &deform_grad,
                                         const ConstitutiveModel<Scalar,Dim> &material)
    :Particle<Scalar,Dim>(pos,vel,mass,vol),F_(deform_grad),constitutive_model_(NULL)
{
    setConstitutiveModel(material);
}

template <typename Scalar, int Dim>
SolidParticle<Scalar,Dim>::SolidParticle(const SolidParticle<Scalar,Dim> &particle)
    :Particle<Scalar,Dim>(particle), F_(particle.F_),constitutive_model_(NULL)
{
    if(particle.constitutive_model_)
        setConstitutiveModel(*(particle.constitutive_model_));
    else
        constitutive_model_ = NULL;
}

template <typename Scalar, int Dim>
SolidParticle<Scalar,Dim>::~SolidParticle()
{
    if(constitutive_model_)
        delete constitutive_model_;
}

template <typename Scalar, int Dim>
SolidParticle<Scalar,Dim>* SolidParticle<Scalar,Dim>::clone() const
{
    return new SolidParticle<Scalar,Dim>(*this);
}

template <typename Scalar, int Dim>
SolidParticle<Scalar,Dim>& SolidParticle<Scalar,Dim>::operator= (const SolidParticle<Scalar,Dim> &particle)
{
    Particle<Scalar,Dim>::operator= (particle);
    F_ = particle.F_;
    if(particle.constitutive_model_)
        setConstitutiveModel(*(particle.constitutive_model_));
    else
        constitutive_model_ = NULL;
    return *this;
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> SolidParticle<Scalar,Dim>::deformationGradient() const
{
    return F_;
}

template <typename Scalar, int Dim>
const ConstitutiveModel<Scalar,Dim>& SolidParticle<Scalar,Dim>::constitutiveModel() const
{
    if(constitutive_model_==NULL)
        throw PhysikaException("SolidParticle constitutive model not set!");
    return *constitutive_model_;
}

template <typename Scalar, int Dim>
ConstitutiveModel<Scalar,Dim>& SolidParticle<Scalar,Dim>::constitutiveModel()
{
    if(constitutive_model_==NULL)
        throw PhysikaException("SolidParticle constitutive model not set!");
    return *constitutive_model_;
}
    
template <typename Scalar, int Dim>
Scalar SolidParticle<Scalar,Dim>::energy() const
{
    if(constitutive_model_==NULL)
        throw PhysikaException("SolidParticle constitutive model not set!");
    return constitutive_model_->energyDensity(F_)*this->volume();
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> SolidParticle<Scalar,Dim>::firstPiolaKirchhoffStress() const
{
    if(constitutive_model_==NULL)
        throw PhysikaException("SolidParticle constitutive model not set!");
    return constitutive_model_->firstPiolaKirchhoffStress(F_);
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> SolidParticle<Scalar,Dim>::secondPiolaKirchhoffStress() const
{
    if(constitutive_model_==NULL)
        throw PhysikaException("SolidParticle constitutive model not set!");
    return constitutive_model_->secondPiolaKirchhoffStress(F_);
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> SolidParticle<Scalar,Dim>::cauchyStress() const
{
    if(constitutive_model_==NULL)
        throw PhysikaException("SolidParticle constitutive model not set!");
    return constitutive_model_->cauchyStress(F_);
}

template <typename Scalar, int Dim>
void SolidParticle<Scalar,Dim>::setDeformationGradient(const SquareMatrix<Scalar,Dim> &F)
{
    F_ = F;
}

template <typename Scalar, int Dim>
void SolidParticle<Scalar,Dim>::setConstitutiveModel(const ConstitutiveModel<Scalar,Dim> &material)
{
    if(constitutive_model_)
        delete constitutive_model_;
    constitutive_model_ = material.clone();
}

//explicit instantiations
template class SolidParticle<float,2>;
template class SolidParticle<float,3>;
template class SolidParticle<double,2>;
template class SolidParticle<double,3>;

} // end of namespace Physika
