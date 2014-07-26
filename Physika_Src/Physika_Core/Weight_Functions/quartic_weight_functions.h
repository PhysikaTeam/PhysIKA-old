/*
 * @file quartic_weight_functions.h 
 * @brief collection of quartic weight functions. 
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

#ifndef PHYSIKA_CORE_WEIGHT_FUNCTIONS_QUARTIC_WEIGHT_FUNCTIONS_H_
#define PHYSIKA_CORE_WEIGHT_FUNCTIONS_QUARTIC_WEIGHT_FUNCTIONS_H_

#include "Physika_Core/Weight_Functions/weight_function.h"

namespace Physika{

/*
 * LucyQuarticWeightFunction:
 * reference: "Numerical approach to testing the fission hypothesis"
 * f(r) = a*(1+3*(r/R))*(1-r/R)^3,  0 <= r <= R
 * where 'a' depends on the dimension and radius of support domain
 */

template <typename Scalar, int Dim>
class LucyQuarticWeightFunction: public WeightFunction<Scalar,Dim>
{
public:
    LucyQuarticWeightFunction(){}
    ~LucyQuarticWeightFunction(){}
    Scalar weight(Scalar r, Scalar R) const;
    Scalar gradient(Scalar r, Scalar R) const;
    void printInfo() const;
};

/*
 * NewQuarticWeightFunction:
 * reference: "A general approach for constructing smoothing function for meshfree methods"
 * let R = 2*h
 * f(r) = a*(2/3-9/8*(r/h)^2+19/24*(r/h)^3-5/32*(r/h)^4) (0 <= r <= 2h)
 * where 'a' depends on the dimension and radius of support domain
 */

template <typename Scalar, int Dim>
class NewQuarticWeightFunction: public WeightFunction<Scalar,Dim>
{
public:
    NewQuarticWeightFunction(){}
    ~NewQuarticWeightFunction(){}
    Scalar weight(Scalar r, Scalar R) const;
    Scalar gradient(Scalar r, Scalar R) const;
    void printInfo() const;
};

}  //end of namespace Physika

#endif //PHYSIKA_CORE_WEIGHT_FUNCTIONS_QUARTIC_WEIGHT_FUNCTIONS_H_
