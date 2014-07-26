/*
 * @file quadratic_weight_functions.cpp 
 * @brief collection of quadratic weight functions. 
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
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Weight_Functions/quadratic_weight_functions.h"

namespace Physika{

template <typename Scalar, int Dim>
Scalar JohnsonQuadraticWeightFunction<Scalar,Dim>::weight(Scalar r, Scalar R) const
{
    PHYSIKA_ASSERT(r >= 0);
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    Scalar h = 0.5*R;
    switch(Dim)
    {
    case 1:
        a = 1.0/h;
        break;
    case 2:
        a = 2.0/(PI*h*h);
        break;
    case 3:
        a = 5.0/(4*PI*h*h*h);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar s = r/h;
    return (s>2) ? 0 : a*(3.0/16.0*s*s-3.0/4.0*s+3.0/4.0);
}

template <typename Scalar, int Dim>
Scalar JohnsonQuadraticWeightFunction<Scalar,Dim>::gradient(Scalar r, Scalar R) const
{
    PHYSIKA_ASSERT(r >= 0);
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    Scalar h = 0.5*R;
    switch(Dim)
    {
    case 1:
        a = 1.0/h;
        break;
    case 2:
        a = 2.0/(PI*h*h);
        break;
    case 3:
        a = 5.0/(4.0*PI*h*h*h);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar s = r/h;
    return (s>2) ? 0 : a*(3.0/8.0*s*(1.0/h)-3.0/4.0*(1.0/h));
}

template <typename Scalar, int Dim>
void JohnsonQuadraticWeightFunction<Scalar,Dim>::printInfo() const
{
    std::cout<<"JohnsonQuadratic weight function with support radius R = 2h:\n";
    switch(Dim)
    {
    case 1:
        std::cout<<"f(r) = 1/h*(3/16*(r/h)^2-3/4*(r/h)+3/4) (0<=r<=2h)\n";
        break;
    case 2:
        std::cout<<"f(r) = 2/(PI*h^2)*(3/16*(r/h)^2-3/4*(r/h)+3/4) (0<=r<=2h)\n";
        break;
    case 3:
        std::cout<<"f(r) = 5/(4*PI*h^3)*(3/16*(r/h)^2-3/4*(r/h)+3/4) (0<=r<=2h)\n";
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
}

//explicit instantiations
template class JohnsonQuadraticWeightFunction<float,1>;
template class JohnsonQuadraticWeightFunction<double,1>;
template class JohnsonQuadraticWeightFunction<float,2>;
template class JohnsonQuadraticWeightFunction<double,2>;
template class JohnsonQuadraticWeightFunction<float,3>;
template class JohnsonQuadraticWeightFunction<double,3>;


template <typename Scalar, int Dim>
Scalar DomeShapedQuadraticWeightFunction<Scalar,Dim>::weight(Scalar r, Scalar R) const
{
    PHYSIKA_ASSERT(r >= 0);
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    switch(Dim)
    {
    case 1:
        a = 3.0/(4.0*R);
        break;
    case 2:
        a = 2.0/(PI*R*R);
        break;
    case 3:
        a = 15.0/(8.0*PI*R*R*R);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    return (r>R) ? 0 : a*(1-(r/R)*(r/R));
}

template <typename Scalar, int Dim>
Scalar DomeShapedQuadraticWeightFunction<Scalar,Dim>::gradient(Scalar r, Scalar R) const
{
    PHYSIKA_ASSERT(r >= 0);
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    switch(Dim)
    {
    case 1:
        a = 3.0/(4.0*R);
        break;
    case 2:
        a = 2.0/(PI*R*R);
        break;
    case 3:
        a = 15.0/(8.0*PI*R*R*R);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    return (r>R) ? 0 : a*(-2)*r/(R*R);
}

template <typename Scalar, int Dim>
void DomeShapedQuadraticWeightFunction<Scalar,Dim>::printInfo() const
{
    std::cout<<"DomeShapedQuadratic weight function with support radius R:\n";
    switch(Dim)
    {
    case 1:
        std::cout<<"f(r) =3/(4*R)*(1-(r/R)^2) (0<=r<=R)\n";
        break;
    case 2:
        std::cout<<"f(r) = 2/(PI*R^2)*(1-(r/R)^2) (0<=r<=R)\n";
        break;
    case 3:
        std::cout<<"f(r) = 15/(8*PI*h^3)*(1-(r/R)^2) (0<=r<=R)\n";
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
}

//explicit instantiations
template class DomeShapedQuadraticWeightFunction<float,1>;
template class DomeShapedQuadraticWeightFunction<double,1>;
template class DomeShapedQuadraticWeightFunction<float,2>;
template class DomeShapedQuadraticWeightFunction<double,2>;
template class DomeShapedQuadraticWeightFunction<float,3>;
template class DomeShapedQuadraticWeightFunction<double,3>;

}  //end of namespace Physika
