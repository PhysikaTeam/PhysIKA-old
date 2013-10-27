/*
 * @file  neo_hookean.h
 * @brief Neo-Hookean hyperelastic material model
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
#ifndef PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_NEO_HOOKEAN_H_
#define PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_NEO_HOOKEAN_H_

#include "Physika_Dynamics/Constitutive_Models/constitutive_model.h"

namespace Physika{

class MatrixBase;

template <typename Scalar, int Dim>
class NeoHookean: public ConstitutiveModel
{
public:
    NeoHookean();
    NeoHookean(Scalar lambda, Scalar mu);
    ~NeoHookean();
    void info() const;
    inline Scalar lambda() const{return lambda_;}
    inline void setLambda(Scalar lambda){lambda_=lambda;}
    inline Scalar mu() const{return mu_;}
    inline void setMu(Scalar mu){mu_=mu;}
    Scalar energy(const MatrixBase &F) const;//compute potential energy density with given deformation gradient
    void energyGradient(const MatrixBase &F, MatrixBase &energy_gradient) const;//compute gradient of energy density with respect to deformation gradient
    void energyHessian(const MatrixBase &F, MatrixBase &energy_hessian) const;//compute hessian of energy density with respect to deformation gradient
protected:
    Scalar lambda_;
    Scalar mu_;
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_NEO_HOOKEAN_H_
