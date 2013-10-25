/*
 * @file  neo_hooken.h
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
#ifndef PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_NEO_HOOKEN_H_
#define PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_NEO_HOOKEN_H_

#include "Physika_Core/Matrices/matrix_base.h"
#include "Physika_Dynamics/Constitutive_Models/constitutive_model.h"

namespace Physika{

template <typename Scalar, int Dim>
class NeoHooken: public ConstitutiveModel
{
public:
    NeoHooken();
    NeoHooken(Scalar lambda, Scalar mu);
    ~NeoHooken();
    void info() const;
    inline Scalar lambda() const{return lambda_;}
    inline void setLambda(Scalar lambda){lambda_=lambda;}
    inline Scalar mu() const{return mu_;}
    inline void setMu(Scalar mu){mu_=mu;}
    Scalar energyDensity(const MatrixBase &F) const;//return potential energy density with given deformation gradient
protected:
    Scalar lambda_;
    Scalar mu_;
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_NEO_HOOKEN_H_
