/*
 * @file linear_weight_functions.h 
 * @brief collection of linear weight functions. 
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

#ifndef PHYSIKA_CORE_WEIGHT_FUNCTIONS_LINEAR_WEIGHT_FUNCTIONS_H_
#define PHYSIKA_CORE_WEIGHT_FUNCTIONS_LINEAR_WEIGHT_FUNCTIONS_H_

#include "Physika_Core/Weight_Functions/weight_function.h"

namespace Physika{

/*
 * LinearWeightFunction: the most common linar weight function
 * f(r) = a(R - r) (0<=r<=R), where the value of 'a' depends on
 * the dimension and radius of support domain such that partition
 * of unity is satisfied:
 *   a = 1/(R^2), in 1D
 *   a = 3/(PI*R^3), in 2D
 *   a = 3/(PI*R^4), in 3D 
 */

template <typename Scalar, int Dim>
class LinearWeightFunction: public WeightFunction<Scalar,Dim>
{
public:
    LinearWeightFunction(){}
    ~LinearWeightFunction(){}
    Scalar weight(Scalar r, Scalar R) const;
    Scalar gradient(Scalar r, Scalar R) const;
    void printInfo() const;
};

}  //end of namespace Physika

#endif //PHYSIKA_CORE_WEIGHT_FUNCTIONS_LINEAR_WEIGHT_FUNCTIONS_H_
