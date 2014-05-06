/*
 * @file linear_elasticity.h
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

#ifndef PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_ISOTROPIC_LINEAR_ELASTICITY_H_
#define PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_ISOTROPIC_LINEAR_ELASTICITY_H_

#include "Physika_Dynamics/Constitutive_Models/isotropic_hyperelastic_material.h"

namespace Physika{

template <typename Scalar, int Dim>
class SquareMatrix;

template <typename Scalar, int Dim>
class IsotropicLinearElasticity: public IsotropicHyperelasticMaterial<Scalar,Dim>
{
public:
    IsotropicLinearElasticity();
    //if par_type = YOUNG_AND_POISSON, then: par1 = young's modulus, par2 = poisson_ratio
    //if par_type = LAME_COEFFICIENTS, then: par1 = lambda, par2 = mu
    IsotropicLinearElasticity(Scalar par1, Scalar par2, ModulusType par_type);
    ~IsotropicLinearElasticity();
    void printInfo() const;
    Scalar energy(const SquareMatrix<Scalar,Dim> &F) const;//compute potential energy density from given deformation gradient
    SquareMatrix<Scalar,Dim> firstPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const;
    SquareMatrix<Scalar,Dim> secondPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const;
    SquareMatrix<Scalar,Dim> cauchyStress(const SquareMatrix<Scalar,Dim> &F) const;

protected:
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_ISOTROPIC_LINEAR_ELASTICITY_H_
