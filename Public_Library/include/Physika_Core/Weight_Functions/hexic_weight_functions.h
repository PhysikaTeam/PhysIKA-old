/*
 * @file hexic_weight_functions.h 
 * @brief collection of hexic weight functions. 
 * @author Sheng Yang
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_WEIGHT_FUNCTIONS_HEXIC_WEIGHT_FUNCTIONS_H_
#define PHYSIKA_CORE_WEIGHT_FUNCTIONS_HEXIC_WEIGHT_FUNCTIONS_H_

#include "Physika_Core/Weight_Functions/weight_function.h"

namespace Physika{

/*
 * MullerPoly6Function:
 * Reference: "Particle-Based Fluid Simulation for Interactive Applications"
 * let h = R
 * f(x,R) = a*(h^2 - |x|^2)^3      (0<=|x|<=h)
 * where 'a' depends on the dimension and radius of support domain
 *   a = 35/(32*R^7), in 1D
 *   a = 4/(PI*R^8), in 2D
 *   a = 315/(64*PI*R^9), in 3D 
 */

template <typename Scalar, int Dim>
class MullerPoly6WeightFunction: public WeightFunction<Scalar,Dim>
{
public:
    MullerPoly6WeightFunction(){}
    ~MullerPoly6WeightFunction(){}
    Scalar weight(const Vector<Scalar, Dim> &center_to_x, Scalar R) const;
    Vector<Scalar, Dim> gradient(const Vector<Scalar, Dim> &center_to_x, Scalar R) const;
    Scalar laplacian(const Vector<Scalar,Dim> &center_to_x, Scalar R) const; 
    void printInfo() const;
};

template <typename Scalar>
class MullerPoly6WeightFunction<Scalar, 1>: public WeightFunction<Scalar,1>
{
public:
    MullerPoly6WeightFunction(){}
    ~MullerPoly6WeightFunction(){}
    Scalar weight(Scalar center_to_x, Scalar R) const; 
    Scalar gradient(Scalar center_to_x, Scalar R) const;
    Scalar laplacian(Scalar center_to_x, Scalar R) const;
    void printInfo() const;
};



}  //end of namespace Physika

#endif //PHYSIKA_CORE_WEIGHT_FUNCTIONS_HEXIC_WEIGHT_FUNCTIONS_H_
