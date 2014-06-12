/*
 * @file linear_elasticity.cpp
 * @brief Isotropic linear elastic constitutive model with infinitesimal strain measure
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

#include <cmath>
#include <iostream>
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Dynamics/Constitutive_Models/isotropic_linear_elasticity.h"

namespace Physika{

template <typename Scalar, int Dim>
IsotropicLinearElasticity<Scalar,Dim>::IsotropicLinearElasticity()
{
}

template <typename Scalar, int Dim>
IsotropicLinearElasticity<Scalar,Dim>::IsotropicLinearElasticity(Scalar par1, Scalar par2, typename IsotropicHyperelasticMaterial<Scalar,Dim>::ModulusType par_type)
    :IsotropicHyperelasticMaterial<Scalar,Dim>(par1,par2,par_type)
{
}

template <typename Scalar, int Dim>
IsotropicLinearElasticity<Scalar,Dim>::~IsotropicLinearElasticity()
{
}

template <typename Scalar, int Dim>
void IsotropicLinearElasticity<Scalar,Dim>::printInfo() const
{
    std::cout<<"Isotropic linear elastic material with infinitesimal strain measure:"<<std::endl;
    std::cout<<"Energy density: Psi = 1/2*lambda*(trace(e)^2)+mu*e:e"<<std::endl;
    std::cout<<"Where: Cauchy Strain e = (transpose(F)+F)/2-I, operator ':' is double contraction."<<std::endl;
}

template <typename Scalar, int Dim>
Scalar IsotropicLinearElasticity<Scalar,Dim>::energy(const SquareMatrix<Scalar,Dim> &F) const
{
    SquareMatrix<Scalar,Dim> e = 0.5*(F.transpose()+F)-SquareMatrix<Scalar,Dim>::identityMatrix();
    Scalar trace_e = e.trace();
    Scalar lambda = this->lambda_;
    Scalar mu = this->mu_;
    Scalar energy = 0.5*lambda*trace_e*trace_e+mu*e.doubleContraction(e);
    return energy;
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> IsotropicLinearElasticity<Scalar,Dim>::firstPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const
{
    //for linear elastic materials all stress measures are identical because F ~= I
    return cauchyStress(F);
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> IsotropicLinearElasticity<Scalar,Dim>::secondPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const
{
    //for linear elastic materials all stress measures are identical because F ~= I
    return cauchyStress(F);
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> IsotropicLinearElasticity<Scalar,Dim>::cauchyStress(const SquareMatrix<Scalar,Dim> &F) const
{
    SquareMatrix<Scalar,Dim> identity = SquareMatrix<Scalar,Dim>::identityMatrix();
    SquareMatrix<Scalar,Dim> e = 0.5*(F.transpose()+F)-identity;
    Scalar trace_e = e.trace();
    Scalar lambda = this->lambda_;
    Scalar mu = this->mu_;
    SquareMatrix<Scalar,Dim> stress = lambda*trace_e*identity+2*mu*e;
    return stress;
}

//explicit instantiation of template so that it could be compiled into a lib
template class IsotropicLinearElasticity<float,2>;
template class IsotropicLinearElasticity<double,2>;
template class IsotropicLinearElasticity<float,3>;
template class IsotropicLinearElasticity<double,3>;

} //end of namespace Physika
