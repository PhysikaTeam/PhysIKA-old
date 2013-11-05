/*
 * @file  isotropic_hyperelastic_material.h
 * @brief Abstract parent class of all hyperelastic material models
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

#ifndef PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_ISOTROPIC_HYPERELASTIC_MATERIAL_H_
#define PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_ISOTROPIC_HYPERELASTIC_MATERIAL_H_

#include "Physika_Core/Array/array.h"
#include "Physika_Dynamics/Constitutive_Models/constitutive_model.h"

namespace Physika{

template <typename Scalar, int Dim>
class SquareMatrix;

enum ModulusType{
    YOUNG_AND_POISSON,
    LAME_COEFFICIENTS
};

template <typename Scalar, int Dim>
class IsotropicHyperelasticMaterial: public ConstitutiveModel
{
public:
    IsotropicHyperelasticMaterial(){}
    //if par_type = YOUNG_AND_POISSON, then: par1 = young's modulus, par2 = poisson_ratio
    //if par_type = LAME_COEFFICIENTS, then: par1 = lambda, par2 = mu
    IsotropicHyperelasticMaterial(Scalar par1, Scalar par2, ModulusType par_type);
    virtual ~IsotropicHyperelasticMaterial(){}
    inline Scalar lambda() const{return lambda_;}
    inline void setLambda(Scalar lambda){lambda_=lambda;}
    inline Scalar mu() const{return mu_;}
    inline void setMu(Scalar mu){mu_=mu;}
    Scalar youngsModulus() const;
    void setYoungsModulus(Scalar);
    Scalar poissonRatio() const;
    void setPoissonRatio(Scalar);
    virtual void info() const=0;
    virtual Scalar energy(const SquareMatrix<Scalar,Dim> &F) const=0;//compute potential energy density from given deformation gradient
    virtual SquareMatrix<Scalar,Dim> firstPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const=0;
    virtual SquareMatrix<Scalar,Dim> secondPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const=0;
    virtual SquareMatrix<Scalar,Dim> cauchyStress(const SquareMatrix<Scalar,Dim> &F) const=0;
protected:
    void lameCoefsFromYoungAndPoisson(Scalar young_modulus, Scalar poisson_ratio, Array<Scalar> &lame_coefs) const;
    void youngAndPoissonFromLameCoefs(Scalar lambda, Scalar mu, Array<Scalar> &young_and_poisson) const;
protected:
    //lame constants
    Scalar lambda_;
    Scalar mu_;
};

//implementations
template <typename Scalar, int Dim>
IsotropicHyperelasticMaterial<Scalar,Dim>::IsotropicHyperelasticMaterial(Scalar par1, Scalar par2, ModulusType par_type)
{
    if(par_type == YOUNG_AND_POISSON)
    {
	Array<Scalar> lame_coefs;
	lame_coefs.setSpace(2);
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
Scalar IsotropicHyperelasticMaterial<Scalar,Dim>::youngsModulus() const
{
    Array<Scalar> young_and_poisson;
    young_and_poisson.setSpace(2);
    youngAndPoissonFromLameCoefs(lambda_,mu_,young_and_poisson);
    return young_and_poisson[0];
}

template <typename Scalar, int Dim>
void IsotropicHyperelasticMaterial<Scalar,Dim>::setYoungsModulus(Scalar young_modulus)
{
    Array<Scalar> young_and_poisson;
    young_and_poisson.setSpace(2);
    youngAndPoissonFromLameCoefs(lambda_,mu_,young_and_poisson);
    young_and_poisson[0] = young_modulus;
    Array<Scalar> lame_coefs;
    lame_coefs.setSpace(2);
    lameCoefsFromYoungAndPoisson(young_and_poisson[0],young_and_poisson[1],lame_coefs);
    lambda_ = lame_coefs[0];
    mu_ = lame_coefs[1];
}

template <typename Scalar, int Dim>
Scalar IsotropicHyperelasticMaterial<Scalar,Dim>::poissonRatio() const
{
    Array<Scalar> young_and_poisson;
    young_and_poisson.setSpace(2);
    youngAndPoissonFromLameCoefs(lambda_,mu_,young_and_poisson);
    return young_and_poisson[1];
}

template <typename Scalar, int Dim>
void IsotropicHyperelasticMaterial<Scalar,Dim>::setPoissonRatio(Scalar poisson_ratio)
{
    Array<Scalar> young_and_poisson;
    young_and_poisson.setSpace(2);
    youngAndPoissonFromLameCoefs(lambda_,mu_,young_and_poisson);
    young_and_poisson[1] = poisson_ratio;
    Array<Scalar> lame_coefs;
    lame_coefs.setSpace(2);
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

}//end of namespace Physika

#endif //PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_ISOTROPIC_HYPERELASTIC_MATERIAL_H_
