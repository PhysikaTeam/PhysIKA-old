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

template <typename Scalar>
Scalar PiecewiseCubicSpline<Scalar,1>::weight(Scalar center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar h = 0.5*R;
    Scalar a = 1.0/h;
    Scalar r = abs(center_to_x);
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
Scalar PiecewiseCubicSpline<Scalar,Dim>::weight(const Vector<Scalar,Dim> &center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    Scalar h = 0.5*R;
    switch(Dim)
    {
    case 2:
        a = 15.0/(7*PI*h*h);
        break;
    case 3:
        a = 3.0/(2*PI*h*h*h);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar r = center_to_x.norm();
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

template <typename Scalar>
Scalar PiecewiseCubicSpline<Scalar,1>::gradient(Scalar center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar h = 0.5*R;
    Scalar a = 1.0/h;
    Scalar r = abs(center_to_x);
    Scalar sign = center_to_x>=0 ? 1 : -1;
    Scalar s = r/h;
    if(s>2)
        return 0;
    else if(s>=1)
        return a*(2-s)*(2-s)*(-1)/(2.0*h)*sign;
    else if(s>=0)
        return a*(-2.0*s/h+3.0/2*s*s/h)*sign;
    else
        PHYSIKA_ERROR("r/R must be greater than zero.");
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> PiecewiseCubicSpline<Scalar,Dim>::gradient(const Vector<Scalar,Dim> &center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    Scalar h = 0.5*R;
    switch(Dim)
    {
    case 2:
        a = 15.0/(7*PI*h*h);
        break;
    case 3:
        a = 3.0/(2*PI*h*h*h);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar r = center_to_x.norm();
    Vector<Scalar,Dim> direction = center_to_x / r;
    Scalar s = r/h;
    if(s>2)
        return 0*direction;
    else if(s>=1)
        return a*(2-s)*(2-s)*(-1)/(2.0*h)*direction;
    else if(s>=0)
        return a*(-2.0*s/h+3.0/2*s*s/h)*direction;
    else
        PHYSIKA_ERROR("r/R must be greater than zero.");
}

template <typename Scalar>
Scalar PiecewiseCubicSpline<Scalar,1>::laplacian(Scalar center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar h = 0.5*R;
    Scalar a = 1.0/h;
    Scalar r = abs(center_to_x);
    Scalar s = r/h;
    if(s>2)
        return 0;
    else if(s>=1)
        return a/(-2*h)*(4*(r-center_to_x*center_to_x/r)/(r*r)-4/h+(r+center_to_x*center_to_x/r)/(h*h));
    else if(s>=0)
        return a*(-2/(h*h)+3/(2*h*h*h)*center_to_x*center_to_x/r+3/(2*h*h*h)*r);
    else
        PHYSIKA_ERROR("r/R must be greater than zero.");
}

template <typename Scalar, int Dim>
Scalar PiecewiseCubicSpline<Scalar,Dim>::laplacian(const Vector<Scalar,Dim> &center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    Scalar h = 0.5*R;
    switch(Dim)
    {
    case 2:
        a = 15.0/(7*PI*h*h);
        break;
    case 3:
        a = 3.0/(2*PI*h*h*h);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar r = center_to_x.norm();
    Scalar s = r/h;
    if(s>2)
        return 0;
    else if(s>=1)
    {
        Scalar result = 0;
        for(unsigned int i = 0; i < Dim; ++i)
            result += a/(-2*h)*(4*(r-center_to_x[i]*center_to_x[i]/r)/(r*r)-4/h+(r+center_to_x[i]*center_to_x[i]/r)/(h*h));
        return result;
    }
    else if(s>=0)
    {
        Scalar result = 0;
        for(unsigned int i = 0; i < Dim; ++i)
            result += a*(-2/(h*h)+3/(2*h*h*h)*center_to_x[i]*center_to_x[i]/r+3/(2*h*h*h)*r);
        return result;
    }
    else
        PHYSIKA_ERROR("r/R must be greater than zero.");
}

template <typename Scalar>
void PiecewiseCubicSpline<Scalar,1>::printInfo() const
{
    std::cout<<"Piecewise cubic spline with support radius R = 2h: \n";
    std::cout<<"f(x,R) = 1/h*(2/3-(|x|/h)^2+1/2*(|x|/h)^3) (0<=|x|<=h)\n";
    std::cout<<"f(x,R) = 1/h*(2-(|x|/h))^3/6 (h<=|x|<=2h)\n";
}

template <typename Scalar, int Dim>
void PiecewiseCubicSpline<Scalar,Dim>::printInfo() const
{
    std::cout<<"Piecewise cubic spline with support radius R = 2h: \n";
    switch(Dim)
    {
    case 2:
        std::cout<<"f(x,R) = 15/(7*PI*h^2)*(2/3-(|x|/h)^2+1/2*(|x|/h)^3) (0<=|x|<=h)\n";
        std::cout<<"f(x,R) = 15/(7*PI*h^2)*(2-(|x|/h))^3/6 (h<=|x|<=2h)\n";
        break;
    case 3:
        std::cout<<"f(x,R) = 3/(2*PI*h^3)*(2/3-(|x|/h)^2+1/2*(|x|/h)^3) (0<=|x|<=h)\n";
        std::cout<<"f(x,R) = 3/(2*PI*h^3)*(2-(|x|/h))^3/6 (h<=|x|<=2h)\n";
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

template <typename Scalar>
Scalar DesbrunSpikyWeightFunction<Scalar,1>::weight(Scalar center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 2.0/(pow(R,4));
    Scalar r = abs(center_to_x);
    return a*pow(R-r, 3);
}

template <typename Scalar, int Dim>
Scalar DesbrunSpikyWeightFunction<Scalar,Dim>::weight(const Vector<Scalar,Dim> &center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    switch(Dim)
    {
    case 2:
        a = 10.0/(PI*pow(R,5));
        break;
    case 3:
        a = 15.0/(PI*pow(R,6));
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar r = center_to_x.norm();
    return (r>R) ? 0 : a*pow((R - r),3);
}

template <typename Scalar>
Scalar DesbrunSpikyWeightFunction<Scalar,1>::gradient(Scalar center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 2.0/(pow(R,4));
    Scalar r = abs(center_to_x);
    return (r>R) ? 0 : a*(-3.0)*pow((R - r),2);
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> DesbrunSpikyWeightFunction<Scalar,Dim>::gradient(const Vector<Scalar,Dim> &center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    switch(Dim)
    {
    case 2:
        a = 10.0/(PI*pow(R,5));
        break;
    case 3:
        a = 15.0/(PI*pow(R,6));
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar r = center_to_x.norm();
    Vector<Scalar,Dim> direction = center_to_x/r; 
    return (r>R) ? 0*direction : a*(-3.0)*pow((R - r),2)*direction;
}

template <typename Scalar>
Scalar DesbrunSpikyWeightFunction<Scalar,1>::laplacian(Scalar center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 2.0/(pow(R,4));
    Scalar r = abs(center_to_x);
    return (r>R) ? 0 : a*(6.0)*(R - r);
}

template <typename Scalar, int Dim>
Scalar DesbrunSpikyWeightFunction<Scalar,Dim>::laplacian(const Vector<Scalar,Dim> &center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    switch(Dim)
    {
    case 2:
        a = 10.0/(PI*pow(R,5));
        break;
    case 3:
        a = 15.0/(PI*pow(R,6));
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar r = center_to_x.norm();  
    if(r > R)
        return 0;
    Scalar result = 0;
    for(unsigned int i = 0; i < Dim; ++i)
        result += 3.0*a*(center_to_x[i]*center_to_x[i]*(R*R/pow(r,3) - 1.0/r) - (R*R/pow(r,3) -2.0*R + r));
    return result;
    
}

template <typename Scalar>
void DesbrunSpikyWeightFunction<Scalar,1>::printInfo() const
{
    std::cout<<"Desbrun Spiky Weight Function with support radius R = h: \n";
    std::cout<<"f(x,R) = 2/h^4*(h - r)^3 (0<=|x|<=h)\n";
}

template <typename Scalar, int Dim>
void DesbrunSpikyWeightFunction<Scalar,Dim>::printInfo() const
{
    std::cout<<"Desbrun Spiky Weight Function with support radius R = h: \n";
    switch(Dim)
    {
    case 2:
         std::cout<<"f(x,R) = 10/h^5*(h - r)^3 (0<=|x|<=h)\n";
        break;
    case 3:
         std::cout<<"f(x,R) = 15/h^6*(h - r)^3 (0<=|x|<=h)\n";
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
}

//explicit instantiations
template class DesbrunSpikyWeightFunction<float,1>;
template class DesbrunSpikyWeightFunction<double,1>;
template class DesbrunSpikyWeightFunction<float,2>;
template class DesbrunSpikyWeightFunction<double,2>;
template class DesbrunSpikyWeightFunction<float,3>;
template class DesbrunSpikyWeightFunction<double,3>;



}  //end of namespace Physika
