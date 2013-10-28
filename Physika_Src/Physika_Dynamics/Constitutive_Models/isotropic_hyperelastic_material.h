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

#include "Physika_Dynamics/Constitutive_Models/constitutive_model.h"

namespace Physika{

template <typename Scalar, int Dim>
class SquareMatrix;

template <typename Scalar, int Dim>
class IsotropicHyperelasticMaterial: public ConstitutiveModel
{
public:
    IsotropicHyperelasticMaterial(){}
    IsotropicHyperelasticMaterial(Scalar lambda, Scalar mu):lambda_(lambda),mu_(mu){}
    ~IsotropicHyperelasticMaterial(){}
    inline Scalar lambda() const{return lambda_;}
    inline void setLambda(Scalar lambda){lambda_=lambda;}
    inline Scalar mu() const{return mu_;}
    inline void setMu(Scalar mu){mu_=mu;}
    virtual void info() const=0;
    virtual Scalar energy(const SquareMatrix<Scalar,Dim> &F) const=0;//compute potential energy density from given deformation gradient
    virtual SquareMatrix<Scalar,Dim> firstPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const=0;
    virtual SquareMatrix<Scalar,Dim> secondPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const=0;
    virtual SquareMatrix<Scalar,Dim> cauchyStress(const SquareMatrix<Scalar,Dim> &F) const=0;

protected:
    //lame constants
    Scalar lambda_;
    Scalar mu_;
};

}//end of namespace Physika

#endif //PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_ISOTROPIC_HYPERELASTIC_MATERIAL_H_
