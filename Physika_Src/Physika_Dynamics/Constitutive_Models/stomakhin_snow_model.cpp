/*
 * @file stomakhin_snow_model.cpp
 * @brief the isotropic elastoplastic model Stomakhin et al. presented in their SIGGRAPH paper
 *            "A material point method for snow simulation"
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

#include <limits>
#include <iostream>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Dynamics/Constitutive_Models/stomakhin_snow_model.h"

namespace Physika{

template <typename Scalar, int Dim>
StomakhinSnowModel<Scalar,Dim>::StomakhinSnowModel()
    :IsotropicHyperelasticMaterial<Scalar,Dim>(),
    stretching_yield_((std::numeric_limits<Scalar>::max)()),
    compression_yield_(std::numeric_limits<Scalar>::lowest()),
    hardening_factor_(0.0)
{
}

template <typename Scalar, int Dim>
StomakhinSnowModel<Scalar,Dim>::StomakhinSnowModel(Scalar par1, Scalar par2,
    typename IsotropicHyperelasticMaterialInternal::ModulusType par_type)
    :IsotropicHyperelasticMaterial<Scalar,Dim>(par1,par2,par_type),
    stretching_yield_((std::numeric_limits<Scalar>::max)()),
    compression_yield_(std::numeric_limits<Scalar>::lowest()),
    hardening_factor_(0.0)
{
}

template <typename Scalar, int Dim>
StomakhinSnowModel<Scalar,Dim>::StomakhinSnowModel(Scalar par1, Scalar par2,
    typename IsotropicHyperelasticMaterialInternal::ModulusType par_type,
    Scalar stretching_yield, Scalar compression_yield, Scalar hardening_factor)
    :IsotropicHyperelasticMaterial<Scalar,Dim>(par1,par2,par_type),
    stretching_yield_(stretching_yield),
    compression_yield_(compression_yield),
    hardening_factor_(hardening_factor)
{
}

template <typename Scalar, int Dim>
StomakhinSnowModel<Scalar,Dim>::StomakhinSnowModel(const StomakhinSnowModel<Scalar,Dim> &material)
    :IsotropicHyperelasticMaterial<Scalar,Dim>(material),
    stretching_yield_(material.stretching_yield_),
    compression_yield_(material.compression_yield_),
    hardening_factor_(material.hardening_factor_)
{
}

template <typename Scalar, int Dim>
StomakhinSnowModel<Scalar,Dim>::~StomakhinSnowModel()
{
}

template <typename Scalar, int Dim>
StomakhinSnowModel<Scalar,Dim>& StomakhinSnowModel<Scalar,Dim>::operator= (const StomakhinSnowModel<Scalar,Dim> &material)
{
    this->mu_ = material.mu_;
    this->lambda_ = material.lambda_;
    this->stretching_yield_ = material.stretching_yield_;
    this->compression_yield_ = material.compression_yield_;
    this->hardening_factor_ = material.hardening_factor_;
    return *this;
}

template <typename Scalar, int Dim>
StomakhinSnowModel<Scalar,Dim>* StomakhinSnowModel<Scalar,Dim>::clone() const
{
    return new StomakhinSnowModel<Scalar,Dim>(*this);
}

template <typename Scalar, int Dim>
Scalar StomakhinSnowModel<Scalar,Dim>::stretchingYield() const
{
    return stretching_yield_;
}

template <typename Scalar, int Dim>
void StomakhinSnowModel<Scalar,Dim>::setStretchingYield(Scalar stretching_yield)
{
    stretching_yield_ = stretching_yield;
}

template <typename Scalar, int Dim>
Scalar StomakhinSnowModel<Scalar,Dim>::compressionYield() const
{
    return compression_yield_;
}

template <typename Scalar, int Dim>
void StomakhinSnowModel<Scalar,Dim>::setCompressionYield(Scalar compression_yield)
{
    compression_yield_ = compression_yield;
}

template <typename Scalar, int Dim>
Scalar StomakhinSnowModel<Scalar,Dim>::hardeningFactor() const
{
    return hardening_factor_;
}

template <typename Scalar, int Dim>
void StomakhinSnowModel<Scalar,Dim>::setHardeningFactor(Scalar hardening_factor)
{
    hardening_factor_ = hardening_factor;
}

template <typename Scalar, int Dim>
void StomakhinSnowModel<Scalar,Dim>::printInfo() const
{
    std::cout<<"StomakhinSnowModel: \n";
    std::cout<<"Energy density: Psi(F_e,F_p) = mu*||F_e-R_e||^2+1/2*lambda*(J_e-1)^2\n";
    std::cout<<"Where: mu and lambda varies with deformation.\n";
    std::cout<<"Reference: A material point method for snow simulation.\n";
}

template <typename Scalar, int Dim>
Scalar StomakhinSnowModel<Scalar,Dim>::energyDensity(const SquareMatrix<Scalar,Dim> &F) const
{
    SquareMatrix<Scalar,Dim> F_e,R_e,F_p;
    Scalar lambda,mu;
    prepareParameters(F,F_e,R_e,F_p,lambda,mu);
    Scalar norm = (F_e-R_e).frobeniusNorm();
    Scalar J_e = F_e.determinant();
    Scalar energy = mu*norm*norm+0.5*lambda*(J_e-1)*(J_e-1);
    return energy;
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> StomakhinSnowModel<Scalar,Dim>::firstPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const
{
    SquareMatrix<Scalar,Dim> F_e,R_e,F_p;
    Scalar lambda,mu;
    prepareParameters(F,F_e,R_e,F_p,lambda,mu);
    Scalar J_e = F_e.determinant();
    return 2*mu*(F_e-R_e)+lambda*(J_e-1)*J_e*(F_e.inverse()).transpose();
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> StomakhinSnowModel<Scalar,Dim>::secondPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const
{
    SquareMatrix<Scalar,Dim> F_e,R_e,F_p;
    Scalar lambda,mu;
    prepareParameters(F,F_e,R_e,F_p,lambda,mu);
    Scalar J_e = F_e.determinant();
    SquareMatrix<Scalar,Dim> F_e_inverse = F_e.inverse();
    SquareMatrix<Scalar,Dim> P = 2*mu*(F_e-R_e)+lambda*(J_e-1)*J_e*F_e_inverse.transpose();
    return F_e_inverse*P;
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> StomakhinSnowModel<Scalar,Dim>::cauchyStress(const SquareMatrix<Scalar,Dim> &F) const
{
    SquareMatrix<Scalar,Dim> F_e,R_e,F_p;
    Scalar lambda,mu;
    prepareParameters(F,F_e,R_e,F_p,lambda,mu);
    Scalar J_e = F_e.determinant();
    return 2*mu/J_e*(F_e-R_e)*F_e.transpose()+lambda/J_e*(J_e-1)*J_e*SquareMatrix<Scalar,Dim>::identityMatrix();
}

template <typename Scalar, int Dim>
void StomakhinSnowModel<Scalar,Dim>::prepareParameters(const SquareMatrix<Scalar,Dim> &F,
                                                        SquareMatrix<Scalar,Dim> &F_e,
                                                        SquareMatrix<Scalar,Dim> &R_e,
                                                        SquareMatrix<Scalar,Dim> &F_p,
                                                        Scalar &lambda, Scalar &mu) const
{
    SquareMatrix<Scalar,Dim> U_e,V_e;
    SquareMatrix<Scalar,Dim> S_e;
    F.singularValueDecomposition(U_e,S_e,V_e);
    for(unsigned int i = 0; i < Dim; ++i)
    {
        if(S_e(i,i) > stretching_yield_)
            S_e(i,i) = stretching_yield_;
        if(S_e(i,i) < compression_yield_)
            S_e(i,i) = compression_yield_;
    }
    F_p = V_e*S_e.inverse()*U_e.transpose()*F;
    F_e = U_e*S_e*V_e.transpose();
    R_e = U_e*V_e.transpose();
    Scalar hardening_power = std::pow(E,hardening_factor_*(1.0-F_p.determinant()));
    lambda = hardening_power * (this->lambda_);
    mu = hardening_power * (this->mu_);
}

//explicit instantiations
template class StomakhinSnowModel<float,2>;
template class StomakhinSnowModel<float,3>;
template class StomakhinSnowModel<double,2>;
template class StomakhinSnowModel<double,3>;

} //end of namespace Physika
