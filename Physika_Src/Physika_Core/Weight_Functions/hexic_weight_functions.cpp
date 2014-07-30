/*
 * @file hexic_weight_functions.h 
 * @brief collection of hexic weight functions. 
 * @author Sheng Yang
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
#include "Physika_Core/Weight_Functions/hexic_weight_functions.h"

namespace Physika{

template <typename Scalar>
Scalar MullerPoly6WeightFunction<Scalar, 1>::weight(Scalar center_to_x, Scalar R) const
{
     PHYSIKA_ASSERT(R > 0);
     Scalar a = 35.0/(32.0*pow(R,7));
     Scalar r = abs(center_to_x);
     return (r>R) ? 0 : a*pow((R*R-r*r),3);
}

template <typename Scalar, int Dim>
Scalar MullerPoly6WeightFunction<Scalar,Dim>::weight(const Vector<Scalar, Dim> &center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    switch(Dim)
    {
    case 2:
        a = 4.0/(PI*pow(R,8));
        break;
    case 3:
        a = 315.0/(64.0*PI*pow(R,9));
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar r = center_to_x.norm();
    return (r>R) ? 0 : a*pow((R*R-r*r),3);
}

template <typename Scalar>
Scalar MullerPoly6WeightFunction<Scalar, 1>::gradient(Scalar center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 35.0/(32.0*pow(R,7));
    Scalar r = abs(center_to_x);
    return (r>R) ? 0 : a*(-6.0*r)*(R*R-r*r)*(R*R-r*r);
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> MullerPoly6WeightFunction<Scalar,Dim>::gradient(const Vector<Scalar,Dim> &center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    switch(Dim)
    {
    case 2:
        a = 4.0/(PI*pow(R,8));
        break;
    case 3:
        a = 315.0/(64.0*PI*pow(R,9));
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar r = center_to_x.norm();
    Vector<Scalar,Dim> direction = center_to_x/r; 
    return (r>R) ? 0*direction : a*(-6.0*r)*(R*R-r*r)*(R*R-r*r)*direction;
}


template <typename Scalar>
Scalar MullerPoly6WeightFunction<Scalar,1>::laplacian(Scalar center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 35.0/(32.0*pow(R,7));
    Scalar r = abs(center_to_x);
    return (r>R) ? 0 : a*(-6.0)*(R*R-r*r)*(R*R-5*r*r);;
}

template <typename Scalar, int Dim>
Scalar MullerPoly6WeightFunction<Scalar,Dim>::laplacian(const Vector<Scalar,Dim> &center_to_x, Scalar R) const
{
    PHYSIKA_ASSERT(R > 0);
    PHYSIKA_ASSERT(R > 0);
    Scalar a = 1.0;
    switch(Dim)
    {
    case 2:
        a = 4.0/(PI*pow(R,8));
        break;
    case 3:
        a = 315.0/(64.0*PI*pow(R,9));
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    Scalar r = center_to_x.norm();
    if(r > R) return 0;
    Scalar result = 0;
    for(unsigned int i = 0; i < Dim; ++i)
        result += a*(24.0*center_to_x[i]*center_to_x[i]*(R*R - r*r) - 6.0*(R*R - r*r)*(R*R - r*r));

    return result;
}



template <typename Scalar>
void MullerPoly6WeightFunction<Scalar, 1>::printInfo() const
{
    std::cout<<"Muller Poly6 Weight Function with support radius R: \n";
    std::cout<<"f(x,R) = 35.0/(32.0*R^7)*(R*R-|x|*|x|)^3 (0<=|x|<=R)\n";
}

template <typename Scalar, int Dim>
void MullerPoly6WeightFunction<Scalar, Dim>::printInfo() const
{
    std::cout<<"Muller Poly6 Weight Function with support radius R: \n";
    switch(Dim)
    {
    case 2:
        std::cout<<"f(x,R) = 4.0/(PI*R^8)*(R*R-|x|*|x|)^3 (0<=|x|<=R)\n";
        break;
    case 3:
        std::cout<<"f(x,R) = 315.0/(64.0*PI*R^9)*(R*R-|x|*|x|)^3 (0<=|x|<=R)\n";
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
}

//explicit instantiations
template class MullerPoly6WeightFunction<float,1>;
template class MullerPoly6WeightFunction<double,1>;
template class MullerPoly6WeightFunction<float,2>;
template class MullerPoly6WeightFunction<double,2>;
template class MullerPoly6WeightFunction<float,3>;
template class MullerPoly6WeightFunction<double,3>;

}  //end of namespace Physika
