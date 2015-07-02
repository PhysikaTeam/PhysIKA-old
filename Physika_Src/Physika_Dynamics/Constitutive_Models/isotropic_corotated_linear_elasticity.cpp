/*
 * @file isotropic_corotated_linear_elasticity.cpp
 * @brief Corotated version of isotropic linear elastic constitutive model with infinitesimal strain measure
 * @author Fei Zhu
 *
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0.
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <cmath>
#include <iostream>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Dynamics/Constitutive_Models/isotropic_corotated_linear_elasticity.h"

namespace Physika{

template <typename Scalar, int Dim>
IsotropicCorotatedLinearElasticity<Scalar,Dim>::IsotropicCorotatedLinearElasticity()
    :IsotropicHyperelasticMaterial<Scalar,Dim>()
{
}

template <typename Scalar, int Dim>
IsotropicCorotatedLinearElasticity<Scalar,Dim>::IsotropicCorotatedLinearElasticity(Scalar par1, Scalar par2, typename IsotropicHyperelasticMaterialInternal::ModulusType par_type)
    :IsotropicHyperelasticMaterial<Scalar,Dim>(par1,par2,par_type)
{
}

template <typename Scalar, int Dim>
IsotropicCorotatedLinearElasticity<Scalar,Dim>::IsotropicCorotatedLinearElasticity(const IsotropicCorotatedLinearElasticity<Scalar,Dim> &material)
    :IsotropicHyperelasticMaterial<Scalar,Dim>(material)
{
}

template <typename Scalar, int Dim>
IsotropicCorotatedLinearElasticity<Scalar,Dim>::~IsotropicCorotatedLinearElasticity()
{
}

template <typename Scalar, int Dim>
IsotropicCorotatedLinearElasticity<Scalar,Dim>& IsotropicCorotatedLinearElasticity<Scalar,Dim>::operator= (const IsotropicCorotatedLinearElasticity<Scalar,Dim> &material)
{
    this->mu_ = material.mu_;
    this->lambda_ = material.lambda_;
    return *this;
}

template <typename Scalar, int Dim>
IsotropicCorotatedLinearElasticity<Scalar,Dim>* IsotropicCorotatedLinearElasticity<Scalar,Dim>::clone() const
{
    return new IsotropicCorotatedLinearElasticity<Scalar,Dim>(*this);
}

template <typename Scalar, int Dim>
void IsotropicCorotatedLinearElasticity<Scalar,Dim>::printInfo() const
{
    std::cout<<"Isotropic corotated linear elastic material with infinitesimal strain measure:\n";
    std::cout<<"Energy density: Psi = 1/2*lambda*(trace(transpose(R)F-I)^2)+mu*(F-R):(F-R)\n";
    std::cout<<"Where: Polar decomposition of F = RS, operator ':' is double contraction.\n";
    std::cout<<"Equivalent form:\n";
    std::cout<<"Energy density: Psi = 1/2*lambda*(trace(sigma-I)^2)+mu*(sigma-I):(sigma-I)\n";
    std::cout<<"Where: sigma is the diagonal matrix with singular values of F from SVD of F.\n";
}

template <typename Scalar, int Dim>
Scalar IsotropicCorotatedLinearElasticity<Scalar,Dim>::energyDensity(const SquareMatrix<Scalar,Dim> &F) const
{
    SquareMatrix<Scalar,Dim> U,V;
    Vector<Scalar,Dim> sigma;
    F.singularValueDecomposition(U,sigma,V);
    Scalar lambda_term = 0, mu_term = 0;
    for(unsigned int i = 0; i < Dim; ++i)
    {
        lambda_term += sigma[i] - 1;
        mu_term += (sigma[i] - 1)*(sigma[i] -1);
    }
    lambda_term *= lambda_term;
    Scalar lambda = this->lambda_;
    Scalar mu = this->mu_;
    Scalar energy = 0.5*lambda*lambda_term+mu*mu_term;
    return energy;
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> IsotropicCorotatedLinearElasticity<Scalar,Dim>::firstPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const
{
    SquareMatrix<Scalar,Dim> U,V;
    Vector<Scalar,Dim> sigma;
    F.singularValueDecomposition(U,sigma,V);
    SquareMatrix<Scalar,Dim> R = U*V.transpose();
    Scalar lambda = this->lambda_;
    Scalar mu = this->mu_;
    SquareMatrix<Scalar,Dim> identity = SquareMatrix<Scalar,Dim>::identityMatrix();
    return 2*mu*(F-R)+lambda*(R.transpose()*F-identity).trace()*R;
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> IsotropicCorotatedLinearElasticity<Scalar,Dim>::secondPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const
{
    return F.inverse()*firstPiolaKirchhoffStress(F);
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> IsotropicCorotatedLinearElasticity<Scalar,Dim>::cauchyStress(const SquareMatrix<Scalar,Dim> &F) const
{
    Scalar J = F.determinant();
    SquareMatrix<Scalar,Dim> stress = 1/J*firstPiolaKirchhoffStress(F)*F.transpose();
    return stress;
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> IsotropicCorotatedLinearElasticity<Scalar,Dim>::firstPiolaKirchhoffStressDifferential(
                                                                const SquareMatrix<Scalar,Dim> &F,
                                                                const SquareMatrix<Scalar,Dim> &F_differential) const
{
    //TO DO
    return SquareMatrix<Scalar,Dim>(0);
}

//explicit instantiation of template so that it could be compiled into a lib
template class IsotropicCorotatedLinearElasticity<float,2>;
template class IsotropicCorotatedLinearElasticity<double,2>;
template class IsotropicCorotatedLinearElasticity<float,3>;
template class IsotropicCorotatedLinearElasticity<double,3>;

} //end of namespace Physika
