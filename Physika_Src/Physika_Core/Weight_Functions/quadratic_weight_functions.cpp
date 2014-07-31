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

#include <limits>
#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Weight_Functions/quadratic_weight_functions.h"

namespace Physika{

template <typename Scalar>
Scalar JohnsonQuadraticWeightFunction<Scalar,1>::weight(Scalar center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar h = 0.5*R;
    Scalar a = 1.0/h;
    Scalar r = abs(center_to_x);
    Scalar s = r/h;
    return (s>2) ? 0 : a*(3.0/16.0*s*s-3.0/4.0*s+3.0/4.0);
}

template <typename Scalar, int Dim>
Scalar JohnsonQuadraticWeightFunction<Scalar,Dim>::weight(const Vector<Scalar,Dim> &center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    Scalar h = 0.5*R;
    switch(Dim)
    {
    case 2:
        a = 2.0/(PI*h*h);
        break;
    case 3:
        a = 5.0/(4*PI*h*h*h);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar r = center_to_x.norm();
    Scalar s = r/h;
    return (s>2) ? 0 : a*(3.0/16.0*s*s-3.0/4.0*s+3.0/4.0);
}

template <typename Scalar>
Scalar JohnsonQuadraticWeightFunction<Scalar,1>::gradient(Scalar center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar h = 0.5*R;
    Scalar a = 1.0/h;
    Scalar r = abs(center_to_x);
    Scalar s = r/h;
    Scalar sign = center_to_x>=0 ? 1 : -1;
    return (s>2) ? 0 : a*(3.0/8.0*s*(1.0/h)-3.0/4.0*(1.0/h))*sign;
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> JohnsonQuadraticWeightFunction<Scalar,Dim>::gradient(const Vector<Scalar,Dim> &center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    Scalar h = 0.5*R;
    switch(Dim)
    {
    case 2:
        a = 2.0/(PI*h*h);
        break;
    case 3:
        a = 5.0/(4.0*PI*h*h*h);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar r = center_to_x.norm();
    Vector<Scalar,Dim> direction(0);
    if(r>0)
        direction = center_to_x/r;
    else
        direction[0] = 1.0;  //set direction to x axis when x is at center 
    Scalar s = r/h;
    return (s>2) ? 0*direction : a*(3.0/8.0*s*(1.0/h)-3.0/4.0*(1.0/h))*direction;
}

template <typename Scalar>
Scalar JohnsonQuadraticWeightFunction<Scalar,1>::laplacian(Scalar center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar h = 0.5*R;
    Scalar a = 1.0/h;
    Scalar r = abs(center_to_x);
    Scalar s = r/h;
    return (s>2) ? 0 : a*3.0/(8.0*h*h);
}

template <typename Scalar, int Dim>
Scalar JohnsonQuadraticWeightFunction<Scalar,Dim>::laplacian(const Vector<Scalar,Dim> &center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    Scalar h = 0.5*R;
    switch(Dim)
    {
    case 2:
        a = 2.0/(PI*h*h);
        break;
    case 3:
        a = 5.0/(4.0*PI*h*h*h);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar r = center_to_x.norm();
    Scalar s = r/h;
    Scalar result = 0;
    if(s>2)
        result = 0;
    else if(s>0)
        result = a*(3.0/(4.0*h*r)+Dim*(3.0/(8.0*h*h)-3.0/(4.0*h*r)));
    else
        result = std::numeric_limits<Scalar>::max();  //infinite
    return result;
}

template <typename Scalar>
void JohnsonQuadraticWeightFunction<Scalar,1>::printInfo() const
{
    std::cout<<"JohnsonQuadratic weight function with support radius R = 2h:\n";
    std::cout<<"f(x,h) = 1/h*(3/16*(|x|/h)^2-3/4*(|x|/h)+3/4) (0<=|x|<=2h)\n";
}

template <typename Scalar, int Dim>
void JohnsonQuadraticWeightFunction<Scalar,Dim>::printInfo() const
{
    std::cout<<"JohnsonQuadratic weight function with support radius R = 2h:\n";
    switch(Dim)
    {
    case 2:
        std::cout<<"f(x,h) = 2/(PI*h^2)*(3/16*(|x|/h)^2-3/4*(|x|/h)+3/4) (0<=|x|<=2h)\n";
        break;
    case 3:
        std::cout<<"f(x,h) = 5/(4*PI*h^3)*(3/16*(|x|/h)^2-3/4*(|x|/h)+3/4) (0<=|x|<=2h)\n";
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


template <typename Scalar>
Scalar DomeShapedQuadraticWeightFunction<Scalar,1>::weight(Scalar center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 3.0/(4.0*R);
    Scalar r = abs(center_to_x);
    return (r>R) ? 0 : a*(1-(r/R)*(r/R));
}

template <typename Scalar, int Dim>
Scalar DomeShapedQuadraticWeightFunction<Scalar,Dim>::weight(const Vector<Scalar,Dim> &center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    switch(Dim)
    {
    case 2:
        a = 2.0/(PI*R*R);
        break;
    case 3:
        a = 15.0/(8.0*PI*R*R*R);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar r = center_to_x.norm();
    return (r>R) ? 0 : a*(1-(r/R)*(r/R));
}

template <typename Scalar>
Scalar DomeShapedQuadraticWeightFunction<Scalar,1>::gradient(Scalar center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 3.0/(4.0*R);
    Scalar r = abs(center_to_x);
    Scalar sign = center_to_x>=0 ? 1 : -1;
    return (r>R) ? 0 : a*(-2)*r/(R*R)*sign;
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> DomeShapedQuadraticWeightFunction<Scalar,Dim>::gradient(const Vector<Scalar,Dim> &center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    switch(Dim)
    {
    case 2:
        a = 2.0/(PI*R*R);
        break;
    case 3:
        a = 15.0/(8.0*PI*R*R*R);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar r = center_to_x.norm();
    Vector<Scalar,Dim> direction(0);
    if(r>0)
        direction = center_to_x/r;
    else
        direction[0] = 1.0; //set direction to x axis when x is at center
    return (r>R) ? 0*direction : a*(-2)*r/(R*R)*direction;
}

template <typename Scalar>
Scalar DomeShapedQuadraticWeightFunction<Scalar,1>::laplacian(Scalar center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 3.0/(4.0*R);
    Scalar r = abs(center_to_x);
    return (r>R) ? 0 : (-6)*a/(R*R);
}

template <typename Scalar, int Dim>
Scalar DomeShapedQuadraticWeightFunction<Scalar,Dim>::laplacian(const Vector<Scalar,Dim> &center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    switch(Dim)
    {
    case 2:
        a = 2.0/(PI*R*R);
        break;
    case 3:
        a = 15.0/(8.0*PI*R*R*R);
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar r = center_to_x.norm();
    return (r>R) ? 0 : (-6)*a/(R*R);;
}

template <typename Scalar>
void DomeShapedQuadraticWeightFunction<Scalar,1>::printInfo() const
{
    std::cout<<"DomeShapedQuadratic weight function with support radius R:\n";
    std::cout<<"f(x,R) =3/(4*R)*(1-(|x|/R)^2) (0<=|x|<=R)\n";
}

template <typename Scalar, int Dim>
void DomeShapedQuadraticWeightFunction<Scalar,Dim>::printInfo() const
{
    std::cout<<"DomeShapedQuadratic weight function with support radius R:\n";
    switch(Dim)
    {
    case 2:
        std::cout<<"f(x,R) = 2/(PI*R^2)*(1-(|x|/R)^2) (0<=|x|<=R)\n";
        break;
    case 3:
        std::cout<<"f(x,R) = 15/(8*PI*R^3)*(1-(|x|/R)^2) (0<=|x|<=R)\n";
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
