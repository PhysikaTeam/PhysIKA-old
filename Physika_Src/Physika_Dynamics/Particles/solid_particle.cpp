/*
 * @file solid_particle.cpp 
 * @Brief the particle used to represent solid, carry deformation gradient information
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

#include "Physika_Dynamics/Particles/solid_particle.h"

namespace Physika{

template <typename Scalar, int Dim>
SolidParticle<Scalar,Dim>::SolidParticle()
    :Particle<Scalar,Dim>()
{
    F_ = SquareMatrix<Scalar,Dim>::identityMatrix();
}

template <typename Scalar, int Dim>
SolidParticle<Scalar,Dim>::SolidParticle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol)
    :Particle<Scalar,Dim>(pos,vel,mass,vol)
{
    F_ = SquareMatrix<Scalar,Dim>::identityMatrix();
}

template <typename Scalar, int Dim>
SolidParticle<Scalar,Dim>::SolidParticle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol, const SquareMatrix<Scalar,Dim> &deform_grad)
    :Particle<Scalar,Dim>(pos,vel,mass,vol),F_(deform_grad)
{
}

template <typename Scalar, int Dim>
SolidParticle<Scalar,Dim>::SolidParticle(const SolidParticle<Scalar,Dim> &particle)
    :Particle<Scalar,Dim>(particle), F_(particle.F_)
{
}

template <typename Scalar, int Dim>
SolidParticle<Scalar,Dim>::~SolidParticle()
{
}

template <typename Scalar, int Dim>
SolidParticle<Scalar,Dim>& SolidParticle<Scalar,Dim>::operator= (const SolidParticle<Scalar,Dim> &particle)
{
    Particle<Scalar,Dim>::operator= (particle);
    F_ = particle.F_;
    return *this;
}

template <typename Scalar, int Dim>
const SquareMatrix<Scalar,Dim>& SolidParticle<Scalar,Dim>::deformationGradient() const
{
    return F_;
}

template <typename Scalar, int Dim>
void SolidParticle<Scalar,Dim>::setDeformationGradient(const SquareMatrix<Scalar,Dim> &F)
{
    F_ = F;
}

//explicit instantiations
template class SolidParticle<float,2>;
template class SolidParticle<float,3>;
template class SolidParticle<double,2>;
template class SolidParticle<double,3>;

} // end of namespace Physika
