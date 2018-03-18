/*
 * @file linear_weight_functions.h 
 * @brief collection of linear weight functions. 
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

#ifndef PHYSIKA_CORE_WEIGHT_FUNCTIONS_LINEAR_WEIGHT_FUNCTIONS_H_
#define PHYSIKA_CORE_WEIGHT_FUNCTIONS_LINEAR_WEIGHT_FUNCTIONS_H_

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Weight_Functions/weight_function.h"

namespace Physika{

/*
 * LinearWeightFunction: the most common linear weight function
 * f(x,R) = a(1 - |x|/R) (0<=|x|<=R), where the value of 'a' depends on
 * the dimension and radius of support domain such that partition
 * of unity is satisfied:
 *   a = 1/R, in 1D
 *   a = 3/(PI*R^2), in 2D
 *   a = 3/(PI*R^3), in 3D 
 */

template <typename Scalar, int Dim>
class LinearWeightFunction: public WeightFunction<Scalar,Dim>
{
public:
    LinearWeightFunction(){}
    ~LinearWeightFunction(){}
    Scalar weight(const Vector<Scalar,Dim> &center_to_x, Scalar R) const; 
    Vector<Scalar,Dim> gradient(const Vector<Scalar,Dim> &center_to_x, Scalar R) const;
    Scalar laplacian(const Vector<Scalar,Dim> &center_to_x, Scalar R) const;
    void printInfo() const;  
};

template <typename Scalar>
class LinearWeightFunction<Scalar,1>: public WeightFunction<Scalar,1>
{
public:
    LinearWeightFunction(){}
    ~LinearWeightFunction(){}
    Scalar weight(Scalar center_to_x, Scalar R) const; 
    Scalar gradient(Scalar center_to_x, Scalar R) const;
    Scalar laplacian(Scalar center_to_x, Scalar R) const;
    void printInfo() const;  
};

}  //end of namespace Physika

#endif //PHYSIKA_CORE_WEIGHT_FUNCTIONS_LINEAR_WEIGHT_FUNCTIONS_H_
