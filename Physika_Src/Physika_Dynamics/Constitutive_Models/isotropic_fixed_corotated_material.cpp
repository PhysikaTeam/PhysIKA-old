/*
 * @file isotropic_fixed_corotated_material.cpp
 * @brief the corrected version of corotated linear material
 * @reference Energetically Consistent Invertible Elasticity
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
#include "Physika_Dynamics/Constitutive_Models/constitutive_model_internal.h"
#include "Physika_Dynamics/Constitutive_Models/isotropic_fixed_corotated_material.h"

namespace Physika{

template <typename Scalar, int Dim>
IsotropicFixedCorotatedMaterial<Scalar,Dim>::IsotropicFixedCorotatedMaterial()
    :IsotropicHyperelasticMaterial<Scalar,Dim>()
{
}

template <typename Scalar, int Dim>
IsotropicFixedCorotatedMaterial<Scalar,Dim>::IsotropicFixedCorotatedMaterial(Scalar par1, Scalar par2, typename IsotropicHyperelasticMaterialInternal::ModulusType par_type)
    :IsotropicHyperelasticMaterial<Scalar,Dim>(par1,par2,par_type)
{
}

template <typename Scalar, int Dim>
IsotropicFixedCorotatedMaterial<Scalar,Dim>::IsotropicFixedCorotatedMaterial(const IsotropicFixedCorotatedMaterial<Scalar,Dim> &material)
    :IsotropicHyperelasticMaterial<Scalar,Dim>(material)
{
}

template <typename Scalar, int Dim>
IsotropicFixedCorotatedMaterial<Scalar,Dim>::~IsotropicFixedCorotatedMaterial()
{
}

template <typename Scalar, int Dim>
IsotropicFixedCorotatedMaterial<Scalar,Dim>& IsotropicFixedCorotatedMaterial<Scalar,Dim>::operator= (const IsotropicFixedCorotatedMaterial<Scalar,Dim> &material)
{
    this->mu_ = material.mu_;
    this->lambda_ = material.lambda_;
    return *this;
}

template <typename Scalar, int Dim>
IsotropicFixedCorotatedMaterial<Scalar,Dim>* IsotropicFixedCorotatedMaterial<Scalar,Dim>::clone() const
{
    return new IsotropicFixedCorotatedMaterial<Scalar,Dim>(*this);
}

template <typename Scalar, int Dim>
void IsotropicFixedCorotatedMaterial<Scalar,Dim>::printInfo() const
{
    std::cout<<"Isotropic Fixed Corotated Model: \n";
    std::cout<<"Energy density: Psi = mu*||F-R||^2+1/2*lambda*(J-1)^2\n";
    std::cout<<"Where: R is the rotation part of F.\n";
    std::cout<<"Reference: Energetically Consistent Invertible Elasticity.\n";
}

template <typename Scalar, int Dim>
Scalar IsotropicFixedCorotatedMaterial<Scalar,Dim>::energyDensity(const SquareMatrix<Scalar,Dim> &F) const
{
    Scalar mu = this->mu_;
    Scalar lambda = this->lambda_;
    SquareMatrix<Scalar,Dim> U,V;
    Vector<Scalar,Dim> sigma;
    F.singularValueDecomposition(U,sigma,V);
    SquareMatrix<Scalar,Dim> R = U*V.transpose();
    Scalar norm = (F-R).frobeniusNorm();
    Scalar J = F.determinant();
    Scalar energy = mu*norm*norm+0.5*lambda*(J-1)*(J-1);
    return energy;
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> IsotropicFixedCorotatedMaterial<Scalar,Dim>::firstPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const
{
    SquareMatrix<Scalar,Dim> U,V;
    Vector<Scalar,Dim> sigma;
    F.singularValueDecomposition(U,sigma,V);
    SquareMatrix<Scalar,Dim> R = U*V.transpose();
    Scalar lambda = this->lambda_;
    Scalar mu = this->mu_;
    Scalar J = F.determinant();
    return 2*mu*(F-R)+lambda*(J-1)*J*(F.inverse()).transpose();
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> IsotropicFixedCorotatedMaterial<Scalar,Dim>::secondPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const
{
    return F.inverse()*firstPiolaKirchhoffStress(F);
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> IsotropicFixedCorotatedMaterial<Scalar,Dim>::cauchyStress(const SquareMatrix<Scalar,Dim> &F) const
{
    Scalar J = F.determinant();
    SquareMatrix<Scalar,Dim> stress = 1/J*firstPiolaKirchhoffStress(F)*F.transpose();
    return stress;
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> IsotropicFixedCorotatedMaterial<Scalar,Dim>::firstPiolaKirchhoffStressDifferential(
                                                                const SquareMatrix<Scalar,Dim> &F,
                                                                const SquareMatrix<Scalar,Dim> &F_differential) const
{
    SquareMatrix<Scalar,Dim> U,V;
    Vector<Scalar,Dim> sigma;
    F.singularValueDecomposition(U,sigma,V);
    SquareMatrix<Scalar,Dim> R = U*V.transpose();
    Scalar lambda = this->lambda_;
    Scalar mu = this->mu_;
    Scalar J = F.determinant();
    SquareMatrix<Scalar,Dim> R_differential = ConstitutiveModelInternal::rotationDifferential(F,R,F_differential);
    SquareMatrix<Scalar,Dim> J_F_inv_trans = J*(F.inverse()).transpose();
    return 2*mu*(F_differential-R_differential)+lambda*J_F_inv_trans*J_F_inv_trans.doubleContraction(F_differential)
           +lambda*(J-1)*ConstitutiveModelInternal::cofactorMatrixDifferential(F,F_differential);
}

//explicit instantiation of template so that it could be compiled into a lib
template class IsotropicFixedCorotatedMaterial<float,2>;
template class IsotropicFixedCorotatedMaterial<double,2>;
template class IsotropicFixedCorotatedMaterial<float,3>;
template class IsotropicFixedCorotatedMaterial<double,3>;

} //end of namespace Physika
