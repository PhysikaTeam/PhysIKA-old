/*
 * @file mpm_particle.cpp
 * @Brief the particle class used in MPM
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

#include "Physika_Dynamics/MPM/mpm_particle.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMParticle<Scalar,Dim>::MPMParticle()
    :weight_(0)
{
}

template <typename Scalar, int Dim>
MPMParticle<Scalar,Dim>::MPMParticle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol)
    :Particle<Scalar,Dim>(pos,vel,mass,vol)
{
    F_ = SquareMatrix<Scalar,Dim>::identityMatrix();
}

template <typename Scalar, int Dim>
MPMParticle<Scalar,Dim>::MPMParticle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol, const SquareMatrix<Scalar,Dim> &deform_grad, Scalar weight, const Vector<Scalar,Dim> &weight_grad)
    :Particle<Scalar,Dim>(pos,vel,mass,vol),F_(deform_grad),weight_(weight),weight_grad_(weight_grad)
{
}

template <typename Scalar, int Dim>
MPMParticle<Scalar,Dim>::MPMParticle(const MPMParticle<Scalar,Dim> &particle)
    :Particle<Scalar,Dim>(particle), F_(particle.F_),weight_(particle.weight_),weight_grad_(particle.weight_grad_)
{
}

template <typename Scalar, int Dim>
MPMParticle<Scalar,Dim>::~MPMParticle()
{
}

template <typename Scalar, int Dim>
MPMParticle<Scalar,Dim>& MPMParticle<Scalar,Dim>::operator= (const MPMParticle<Scalar,Dim> &particle)
{
    Particle<Scalar,Dim>::operator= (particle);
    F_ = particle.F_;
    weight_ = particle.weight_;
    weight_grad_ = particle.weight_grad_;
    return *this;
}

template <typename Scalar, int Dim>
const SquareMatrix<Scalar,Dim>& MPMParticle<Scalar,Dim>::deformationGradient() const
{
    return F_;
}

template <typename Scalar, int Dim>
void MPMParticle<Scalar,Dim>::setDeformationGradient(const SquareMatrix<Scalar,Dim> &F)
{
    F_ = F;
}

template <typename Scalar, int Dim>
Scalar MPMParticle<Scalar,Dim>::weight() const
{
    return weight_;
}

template <typename Scalar, int Dim>
void MPMParticle<Scalar,Dim>::setWeight(Scalar weight)
{
    weight_ = weight;
}

template <typename Scalar, int Dim>
const Vector<Scalar,Dim>& MPMParticle<Scalar,Dim>::weightGradient() const
{
    return weight_grad_;
}

template <typename Scalar, int Dim>
void MPMParticle<Scalar,Dim>::setWeightGradient(const Vector<Scalar,Dim> &weight_grad)
{
    weight_grad_ = weight_grad;
}

//explicit instantiations
template class MPMParticle<float,2>;
template class MPMParticle<float,3>;
template class MPMParticle<double,2>;
template class MPMParticle<double,3>;

} // end of namespace Physika
