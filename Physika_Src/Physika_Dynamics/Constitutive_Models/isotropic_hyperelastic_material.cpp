/*
 * @file  isotropic_hyperelastic_material.cpp
 * @brief Abstract parent class of all hyperelastic material models
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

#include "Physika_Core/Arrays/array.h"
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Dynamics/Constitutive_Models/isotropic_hyperelastic_material.h"

namespace Physika{

using IsotropicHyperelasticMaterialInternal::ModulusType;
using IsotropicHyperelasticMaterialInternal::YOUNG_AND_POISSON;
using IsotropicHyperelasticMaterialInternal::LAME_COEFFICIENTS;

template <typename Scalar, int Dim>
IsotropicHyperelasticMaterial<Scalar,Dim>::IsotropicHyperelasticMaterial(Scalar par1, Scalar par2, ModulusType par_type)
{
    if(par_type == YOUNG_AND_POISSON)
    {
        Array<Scalar> lame_coefs(2);
        lameCoefsFromYoungAndPoisson(par1,par2,lame_coefs);
        lambda_ = lame_coefs[0];
        mu_ = lame_coefs[1];
    }
    else
    {
        lambda_ = par1;
        mu_ = par2;
    }
}

template <typename Scalar, int Dim>
IsotropicHyperelasticMaterial<Scalar,Dim>::IsotropicHyperelasticMaterial(const IsotropicHyperelasticMaterial<Scalar,Dim> &material)
{
    lambda_ = material.lambda_;
    mu_ = material.mu_;
}

template <typename Scalar, int Dim>
IsotropicHyperelasticMaterial<Scalar,Dim>& IsotropicHyperelasticMaterial<Scalar,Dim>::operator= (const IsotropicHyperelasticMaterial<Scalar,Dim> &material)
{
    lambda_ = material.lambda_;
    mu_ = material.mu_;
    return *this;
}

template <typename Scalar, int Dim>
Scalar IsotropicHyperelasticMaterial<Scalar,Dim>::youngsModulus() const
{
    Array<Scalar> young_and_poisson(2);
    youngAndPoissonFromLameCoefs(lambda_,mu_,young_and_poisson);
    return young_and_poisson[0];
}

template <typename Scalar, int Dim>
void IsotropicHyperelasticMaterial<Scalar,Dim>::setYoungsModulus(Scalar young_modulus)
{
    Array<Scalar> young_and_poisson(2);
    youngAndPoissonFromLameCoefs(lambda_,mu_,young_and_poisson);
    young_and_poisson[0] = young_modulus;
    Array<Scalar> lame_coefs(2);
    lameCoefsFromYoungAndPoisson(young_and_poisson[0],young_and_poisson[1],lame_coefs);
    lambda_ = lame_coefs[0];
    mu_ = lame_coefs[1];
}

template <typename Scalar, int Dim>
Scalar IsotropicHyperelasticMaterial<Scalar,Dim>::poissonRatio() const
{
    Array<Scalar> young_and_poisson(2);
    youngAndPoissonFromLameCoefs(lambda_,mu_,young_and_poisson);
    return young_and_poisson[1];
}

template <typename Scalar, int Dim>
void IsotropicHyperelasticMaterial<Scalar,Dim>::setPoissonRatio(Scalar poisson_ratio)
{
    Array<Scalar> young_and_poisson(2);
    youngAndPoissonFromLameCoefs(lambda_,mu_,young_and_poisson);
    young_and_poisson[1] = poisson_ratio;
    Array<Scalar> lame_coefs(2);
    lameCoefsFromYoungAndPoisson(young_and_poisson[0],young_and_poisson[1],lame_coefs);
    lambda_ = lame_coefs[0];
    mu_ = lame_coefs[1];
}

template <typename Scalar, int Dim>
void IsotropicHyperelasticMaterial<Scalar,Dim>::youngAndPoissonFromLameCoefs(Scalar lambda, Scalar mu, Array<Scalar> &young_and_poisson) const
{
    young_and_poisson[0] = mu_*(3*lambda_+2*mu_)/(lambda_+mu_);
    young_and_poisson[1] = lambda_/(2*(lambda_+mu_));
}

template <typename Scalar, int Dim>
void IsotropicHyperelasticMaterial<Scalar,Dim>::lameCoefsFromYoungAndPoisson(Scalar young_modulus, Scalar poisson_ratio, Array<Scalar> &lame_coefs) const
{
    lame_coefs[0] = (young_modulus*poisson_ratio)/((1+poisson_ratio)*(1-2*poisson_ratio));//lambda
    lame_coefs[1] = young_modulus/(2*(1+poisson_ratio));//mu_
}

//explicit instantiations
template class IsotropicHyperelasticMaterial<float,2>;
template class IsotropicHyperelasticMaterial<float,3>;
template class IsotropicHyperelasticMaterial<double,2>;
template class IsotropicHyperelasticMaterial<double,3>;

}  //end of namespace Physika
