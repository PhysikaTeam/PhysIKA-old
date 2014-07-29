/*
 * @file cubic_weight_functions.h 
 * @brief collection of cubic weight functions. 
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

#ifndef PHYSIKA_CORE_WEIGHT_FUNCTIONS_CUBIC_WEIGHT_FUNCTIONS_H_
#define PHYSIKA_CORE_WEIGHT_FUNCTIONS_CUBIC_WEIGHT_FUNCTIONS_H_

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Weight_Functions/weight_function.h"

namespace Physika{

/*
 * PiecewiseCubicSpline:
 * let h = 0.5*R,
 * f(x,R) = a*(2/3-(|x|/h)^2+1/2*(|x|/h)^3,  0 <= |x| <= h
 * f(x,R) = a*(1/6)*(2-|x|/h)^3,  h <= |x| <= 2h
 * where 'a' depends on the dimension and radius of support domain
 */

template <typename Scalar, int Dim>
class PiecewiseCubicSpline: public WeightFunction<Scalar,Dim>
{
public:
    PiecewiseCubicSpline(){}
    ~PiecewiseCubicSpline(){}
    Scalar weight(const Vector<Scalar,Dim> &center_to_x, Scalar R) const;
    Vector<Scalar,Dim> gradient(const Vector<Scalar,Dim> &center_to_x, Scalar R) const;
    Scalar laplacian(const Vector<Scalar,Dim> &center_to_x, Scalar R) const;
    void printInfo() const;
};

template <typename Scalar>
class PiecewiseCubicSpline<Scalar,1>: public WeightFunction<Scalar,1>
{
public:
    PiecewiseCubicSpline(){}
    ~PiecewiseCubicSpline(){}
    Scalar weight(Scalar center_to_x, Scalar R) const;
    Scalar gradient(Scalar center_to_x, Scalar R) const;
    Scalar laplacian(Scalar center_to_x, Scalar R) const;
    void printInfo() const;
};

}  //end of namespace Physika

#endif //PHYSIKA_CORE_WEIGHT_FUNCTIONS_CUBIC_WEIGHT_FUNCTIONS_H_
