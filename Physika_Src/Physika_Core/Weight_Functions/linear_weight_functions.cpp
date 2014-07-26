/*
 * @file linear_weight_functions.cpp 
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

#include <iostream>
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Weight_Functions/linear_weight_functions.h"

namespace Physika{

template <typename Scalar, int Dim>
Scalar LinearWeightFunction<Scalar,Dim>::weight(Scalar r, Scalar R) const
{
    PHYSIKA_ASSERT(r >= 0);
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    switch(Dim)
    {
    case 1:
        a = 1.0/R;
        break;
    case 2:
        a = 3.0/(PI*R*R);
        break;
    case 3:
        a = 3.0/(PI*R*R*R);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    return (r>R) ? 0 : a*(1-r/R);
}

template <typename Scalar, int Dim>
Scalar LinearWeightFunction<Scalar,Dim>::gradient(Scalar r, Scalar R) const
{
    PHYSIKA_ASSERT(r >= 0);
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    switch(Dim)
    {
    case 1:
        a = 1.0/R;
        break;
    case 2:
        a = 3.0/(PI*R*R);
        break;
    case 3:
        a = 3.0/(PI*R*R*R);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    return (r>R) ? 0 : a*(-1.0/R);
}

template <typename Scalar, int Dim>
void LinearWeightFunction<Scalar,Dim>::printInfo() const
{
    std::cout<<"Linear weight function with support radius R: \n";
    switch(Dim)
    {
    case 1:
        std::cout<<"f(r) = (1/R)*(1-r/R) (0<=r<=R)\n";
        break;
    case 2:
        std::cout<<"f(r) = (3/PI*R^2)*(1-r/R) (0<=r<=R)\n";
        break;
    case 3:
        std::cout<<"f(r) = (3/PI*R^3)*(1-r/R) (0<=r<=R)\n";
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
}

//explicit instantiations
template class LinearWeightFunction<float,1>;
template class LinearWeightFunction<double,1>;
template class LinearWeightFunction<float,2>;
template class LinearWeightFunction<double,2>;
template class LinearWeightFunction<float,3>;
template class LinearWeightFunction<double,3>;

}  //end of namespace Physika
