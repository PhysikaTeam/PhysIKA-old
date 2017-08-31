/*
 * @file linear_weight_functions.cpp 
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

#include <limits>
#include <iostream>
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Weight_Functions/linear_weight_functions.h"

namespace Physika{
template <typename Scalar>
Scalar LinearWeightFunction<Scalar,1>::weight(Scalar center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0/R;
    Scalar r = abs(center_to_x);
    return (r>R) ? 0 : a*(1-r/R);
}

template <typename Scalar, int Dim>
Scalar LinearWeightFunction<Scalar,Dim>::weight(const Vector<Scalar,Dim> &center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    switch(Dim)
    {
    case 2:
        a = 3.0/(PI*R*R);
        break;
    case 3:
        a = 3.0/(PI*R*R*R);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar r = center_to_x.norm();
    return (r>R) ? 0 : a*(1-r/R);
}

template <typename Scalar>
Scalar LinearWeightFunction<Scalar,1>::gradient(Scalar center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0/R;
    Scalar r = abs(center_to_x);
    Scalar sign = center_to_x>=0 ? 1 : -1;
    return (r>R) ? 0 : a*(-1.0/R)*sign;
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> LinearWeightFunction<Scalar,Dim>::gradient(const Vector<Scalar,Dim> &center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    switch(Dim)
    {
    case 2:
        a = 3.0/(PI*R*R);
        break;
    case 3:
        a = 3.0/(PI*R*R*R);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar r = center_to_x.norm();
    Vector<Scalar,Dim> direction(0);
    if(r>std::numeric_limits<Scalar>::epsilon())
        direction = center_to_x/r;
    else
        direction[0] = 1.0;  //set direction to x axis when x is at center 
    return (r>R) ? 0*direction : a*(-1.0/R)*direction;
}

template <typename Scalar>
Scalar LinearWeightFunction<Scalar,1>::laplacian(Scalar center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    return 0;
}

template <typename Scalar, int Dim>
Scalar LinearWeightFunction<Scalar,Dim>::laplacian(const Vector<Scalar,Dim> &center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    switch(Dim)
    {
    case 2:
        a = 3.0/(PI*R*R);
        break;
    case 3:
        a = 3.0/(PI*R*R*R);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar r = center_to_x.norm();
    Scalar result = 0;
    if(r>R)
        result = 0;

    else if(r>std::numeric_limits<Scalar>::epsilon())
        result = (-a/R)*(Dim-1.0)/r;
    else
        result = std::numeric_limits<Scalar>::min();  //infinite
    return result;
}

template <typename Scalar>
void LinearWeightFunction<Scalar,1>::printInfo() const
{
    std::cout<<"Linear weight function with support radius R: \n";
    std::cout<<"f(x,R) = (1/R)*(1-|x|/R) (0<=|x|<=R)\n";
}

template <typename Scalar, int Dim>
void LinearWeightFunction<Scalar,Dim>::printInfo() const
{
    std::cout<<"Linear weight function with support radius R: \n";
    switch(Dim)
    {
    case 2:
        std::cout<<"f(x,R) = (3/PI*R^2)*(1-|x|/R) (0<=|x|<=R)\n";
        break;
    case 3:
        std::cout<<"f(x,R) = (3/PI*R^3)*(1-|x|/R) (0<=|x|<=R)\n";
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
