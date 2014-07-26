/*
 * @file quartic_weight_functions.cpp 
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

#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Weight_Functions/quartic_weight_functions.h"

namespace Physika{

template <typename Scalar, int Dim>
Scalar LucyQuarticWeightFunction<Scalar,Dim>::weight(Scalar r, Scalar R) const
{
    PHYSIKA_ASSERT(r >= 0);
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    switch(Dim)
    {
    case 1:
        a = 5.0/(4.0*R);
        break;
    case 2:
        a = 5.0/(PI*R*R);
        break;
    case 3:
        a = 105.0/(16.0*PI*R*R*R);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    return (r>R) ? 0 : a*(1+3*r/R)*(1-r/R)*(1-r/R)*(1-r/R);
}

template <typename Scalar, int Dim>
Scalar LucyQuarticWeightFunction<Scalar,Dim>::gradient(Scalar r, Scalar R) const
{
    PHYSIKA_ASSERT(r >= 0);
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    switch(Dim)
    {
    case 1:
        a = 5.0/(4.0*R);
        break;
    case 2:
        a = 5.0/(PI*R*R);
        break;
    case 3:
        a = 105.0/(16.0*PI*R*R*R);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    return (r>R) ? 0 : a*((3.0/R)*(1-r/R)*(1-r/R)*(1-r/R)+(1+3*r/R)*3*(1-r/R)*(1-r/R)*(-1.0/R));
}

template <typename Scalar, int Dim>
void LucyQuarticWeightFunction<Scalar,Dim>::printInfo() const
{
    std::cout<<"LucyQuartic weight function with support radius R:\n";
    switch(Dim)
    {
    case 1:
        std::cout<<"f(r) = 5/(4*R)*(1+3*r/R)*(1-r/R)^3 (0<=r<=R)\n";
        break;
    case 2:
        std::cout<<"f(r) = 5/(PI*R^2)*(1+3*r/R)*(1-r/R)^3 (0<=r<=R)\n";
        break;
    case 3:
        std::cout<<"f(r) = 105/(16*PI*R^3)*(1+3*r/R)*(1-r/R)^3 (0<=r<=R)\n";
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
}

//explicit instantiations
template class LucyQuarticWeightFunction<float,1>;
template class LucyQuarticWeightFunction<double,1>;
template class LucyQuarticWeightFunction<float,2>;
template class LucyQuarticWeightFunction<double,2>;
template class LucyQuarticWeightFunction<float,3>;
template class LucyQuarticWeightFunction<double,3>;

template <typename Scalar, int Dim>
Scalar NewQuarticWeightFunction<Scalar,Dim>::weight(Scalar r, Scalar R) const
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
        a = 15.0/(7.0*PI*h*h);
        break;
    case 3:
        a = 315.0/(208.0*PI*h*h*h);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar s = r/h, ss = s*s;
    return (s>2) ? 0 : a*(2.0/3.0-9.0/8.0*ss+19.0/24.0*ss*s-5.0/32.0*ss*ss);
}

template <typename Scalar, int Dim>
Scalar NewQuarticWeightFunction<Scalar,Dim>::gradient(Scalar r, Scalar R) const
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
        a = 15.0/(7.0*PI*h*h);
        break;
    case 3:
        a = 315.0/(208.0*PI*h*h*h);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar s = r/h, ss = s*s;
    return (s>2) ? 0 : a*(-9.0/4.0*s/h+19.0/8.0*ss/h-5.0/8.0*ss*s/h);
}

template <typename Scalar, int Dim>
void NewQuarticWeightFunction<Scalar,Dim>::printInfo() const
{
    std::cout<<"NewQuartic weight function with support radius R = 2h:\n";
    switch(Dim)
    {
    case 1:
        std::cout<<"f(r) = 1/h*(2/3-9/8*(r/h)^2+19/24*(r/h)^3-5/32*(r/h)^4)  (0<=r<=2h)\n";
        break;
    case 2:
        std::cout<<"f(r) = 15/(7*PI*h^2)*(2/3-9/8*(r/h)^2+19/24*(r/h)^3-5/32*(r/h)^4) (0<=r<=2h)\n";
        break;
    case 3:
        std::cout<<"f(r) = 315/(208*PI*h^3)*(2/3-9/8*(r/h)^2+19/24*(r/h)^3-5/32*(r/h)^4) (0<=r<=2h)\n";
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
}

//explicit instantiations
template class NewQuarticWeightFunction<float,1>;
template class NewQuarticWeightFunction<double,1>;
template class NewQuarticWeightFunction<float,2>;
template class NewQuarticWeightFunction<double,2>;
template class NewQuarticWeightFunction<float,3>;
template class NewQuarticWeightFunction<double,3>;

}  //end of namespace Physika
