/*
 * @file cubic_weight_functions.cpp 
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

#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Weight_Functions/cubic_weight_functions.h"

namespace Physika{

template <typename Scalar, int Dim>
Scalar PiecewiseCubicSpline<Scalar,Dim>::weight(Scalar r, Scalar R) const
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
        a = 15.0/(7*PI*h*h);
        break;
    case 3:
        a = 3.0/(2*PI*h*h*h);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar s = r/h;
    if(s>2)
        return 0;
    else if(s>=1)
        return a*(2.0-s)*(2.0-s)*(2.0-s)/6.0;
    else if(s>=0)
        return a*(2.0/3.0-s*s+1.0/2.0*s*s*s);
    else
        PHYSIKA_ERROR("r/R must be greater than zero.");
}

template <typename Scalar, int Dim>
Scalar PiecewiseCubicSpline<Scalar,Dim>::gradient(Scalar r, Scalar R) const
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
        a = 15.0/(7*PI*h*h);
        break;
    case 3:
        a = 3.0/(2*PI*h*h*h);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar s = r/h;
    if(s>2)
        return 0;
    else if(s>=1)
        return a*(2-s)*(2-s)*(-1)/(2.0*h);
    else if(s>=0)
        return a*(-2.0*s/h+3.0/2*s*s/h);
    else
        PHYSIKA_ERROR("r/R must be greater than zero.");
}

template <typename Scalar, int Dim>
void PiecewiseCubicSpline<Scalar,Dim>::printInfo() const
{
    std::cout<<"Piecewise cubic spline with support radius R = 2h: \n";
    switch(Dim)
    {
    case 1:
        std::cout<<"f(r) = 1/h*(2/3-(r/h)^2+1/2*(r/h)^3) (0<=r<=h)\n";
        std::cout<<"f(r) = 1/h*(2-(r/h))^3/6 (h<=r<=2h)\n";
        break;
    case 2:
        std::cout<<"f(r) = 15/(7*PI*h^2)*(2/3-(r/h)^2+1/2*(r/h)^3) (0<=r<=h)\n";
        std::cout<<"f(r) = 15/(7*PI*h^2)*(2-(r/h))^3/6 (h<=r<=2h)\n";
        break;
    case 3:
        std::cout<<"f(r) = 3/(2*PI*h^3)*(2/3-(r/h)^2+1/2*(r/h)^3) (0<=r<=h)\n";
        std::cout<<"f(r) = 3/(2*PI*h^3)*(2-(r/h))^3/6 (h<=r<=2h)\n";
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
}

//explicit instantiations
template class PiecewiseCubicSpline<float,1>;
template class PiecewiseCubicSpline<double,1>;
template class PiecewiseCubicSpline<float,2>;
template class PiecewiseCubicSpline<double,2>;
template class PiecewiseCubicSpline<float,3>;
template class PiecewiseCubicSpline<double,3>;

}  //end of namespace Physika
