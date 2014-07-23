/*
 * @file linear_weight_function.h 
 * @brief linear weight function. 
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

#ifndef PHYSIKA_CORE_WEIGHT_FUNCTIONS_LINEAR_WEIGHT_FUNCTION_H_
#define PHYSIKA_CORE_WEIGHT_FUNCTIONS_LINEAR_WEIGHT_FUNCTION_H_

#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Weight_Functions/weight_function.h"

namespace Physika{

template <typename Scalar>
class LinearWeightFunction: public WeightFunction<Scalar>
{
public:
    LinearWeightFunction(){}
    ~LinearWeightFunction(){}
    Scalar weight(Scalar r, Scalar h) const;
    Scalar gradient(Scalar r, Scalar h) const;
    void printInfo() const;
};

//Implementation
template <typename Scalar>
Scalar LinearWeightFunction<Scalar>::weight(Scalar r, Scalar h) const
{
    PHYSIKA_ASSERT(h>0);
    return (r>h) ? 0 : (h-r)/h;
}

template <typename Scalar>
Scalar LinearWeightFunction<Scalar>::gradient(Scalar r, Scalar h) const
{
    PHYSIKA_ASSERT(h>0);
    return (r>h) ? 0 : -1.0/h;
}

template <typename Scalar>
void LinearWeightFunction<Scalar>::printInfo() const
{
    std::cout<<"Linear weight function: f(r,h) = 1 - r/h (r<=h)\n";
}

}  //end of namespace Physika

#endif //PHYSIKA_CORE_WEIGHT_FUNCTIONS_LINEAR_WEIGHT_FUNCTION_H_
