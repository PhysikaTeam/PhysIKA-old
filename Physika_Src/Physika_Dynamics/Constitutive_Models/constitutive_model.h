/*
 * @file  constitutive_model.h
 * @brief Constitutive model of deformable solids, abstract class
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

#ifndef PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_CONSTITUTIVE_MODEL_H_
#define PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_CONSTITUTIVE_MODEL_H_

#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"

namespace Physika{

template <typename Scalar, int Dim>
class ConstitutiveModel
{
public:
    ConstitutiveModel(){}
    virtual ~ConstitutiveModel(){}
    virtual ConstitutiveModel* clone() const=0;  //clone the constitutive model
    virtual void printInfo() const=0;
    virtual Scalar energyDensity(const SquareMatrix<Scalar,Dim> &F) const=0;//compute potential energy density from given deformation gradient
    virtual SquareMatrix<Scalar,Dim> firstPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const=0;
    virtual SquareMatrix<Scalar,Dim> secondPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const=0;
    virtual SquareMatrix<Scalar,Dim> cauchyStress(const SquareMatrix<Scalar,Dim> &F) const=0;
protected:
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_CONSTITUTIVE_MODEL_H_
