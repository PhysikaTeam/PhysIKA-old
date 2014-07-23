/*
 * @file quadratic_weight_function.h 
 * @brief quadratic weight function. 
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

#ifndef PHYSIKA_CORE_WEIGHT_FUNCTIONS_QUADRATIC_WEIGHT_FUNCTION_H_
#define PHYSIKA_CORE_WEIGHT_FUNCTIONS_QUADRATIC_WEIGHT_FUNCTION_H_

#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Weight_Functions/weight_function.h"

namespace Physika{

template <typename Scalar>
class QuadraticWeightFunction: public WeightFunction<Scalar>
{
public:
    QuadraticWeightFunction(){}
    ~QuadraticWeightFunction(){}
    Scalar weight(Scalar r, Scalar h) const;
    Scalar gradient(Scalar r, Scalar h) const;
    void printInfo() const;
};

//Implementation
template <typename Scalar>
Scalar QuadraticWeightFunction<Scalar>::weight(Scalar r, Scalar h) const
{
    PHYSIKA_ASSERT(h>0);
    Scalar q = r/h;
    if(q>1.0)
        return 0;
    else
    {
        Scalar alpha = 15.0/(2.0*static_cast<Scalar>(PI));
        return alpha*(1.0-q)*(1.0-q);
    }
}

template <typename Scalar>
Scalar QuadraticWeightFunction<Scalar>::gradient(Scalar r, Scalar h) const
{
    PHYSIKA_ASSERT(h>0);
    Scalar q = r/h;
    if(q>1.0)
        return 0;
    else
    {
        Scalar alpha = 15.0/(static_cast<Scalar>(PI));
        return -alpha*(1.0-q);
    }
}

template <typename Scalar>
void QuadraticWeightFunction<Scalar>::printInfo() const
{
    std::cout<<"Quadratic weight function: f(r,h) = 15/(2*PI)*(1-r/h)^2\n";
}

}  //end of namespace Physika

#endif //PHYSIKA_CORE_WEIGHT_FUNCTIONS_QUADRATIC_WEIGHT_FUNCTION_H_
