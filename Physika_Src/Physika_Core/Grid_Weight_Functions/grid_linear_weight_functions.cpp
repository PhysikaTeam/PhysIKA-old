/*
 * @file grid_linear_weight_functions.cpp 
 * @brief collection of linear grid-based weight functions. 
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
#include "Physika_Core/Weight_Functions/linear_weight_functions.h"
#include "Physika_Core/Grid_Weight_Functions/grid_linear_weight_functions.h"

namespace Physika{

template <typename Scalar, int Dim>
Scalar GridLinearWeightFunction<Scalar,Dim>::weight(const Vector<Scalar,Dim> &center_to_x, const Vector<Scalar,Dim> &support_radius) const
{
    LinearWeightFunction<Scalar,1> linear_1d;
    Scalar result = 1.0;
    for(unsigned int i = 0; i < Dim; ++i)
        result *= linear_1d.weight(center_to_x[i],support_radius[i]);
    return result;
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> GridLinearWeightFunction<Scalar,Dim>::gradient(const Vector<Scalar,Dim> &center_to_x, const Vector<Scalar,Dim> &support_radius) const
{
    LinearWeightFunction<Scalar,1> linear_1d;
    Vector<Scalar,Dim> result(1.0);
    for(unsigned int i = 0; i < Dim; ++i)
        for(unsigned int j = 0; j < Dim; ++j)
        {
            if(j==i)
                result[i] *= linear_1d.gradient(center_to_x[j],support_radius[j]);
            else
                result[i] *= linear_1d.weight(center_to_x[j],support_radius[j]);
        }
    return result;
}

template <typename Scalar, int Dim>
void GridLinearWeightFunction<Scalar,Dim>::printInfo() const
{
    switch(Dim)
    {
    case 2:
        std::cout<<"Grid-based linear weight function with support radius (R1,R2):\n";
        std::cout<<"f(x,y,R1,R2) = g(x,R1)*g(y,R2) (0<=|x|<=R1, 0<=|y|<=R2)\n";
        break;
    case 3:
        std::cout<<"Grid-based linear weight function with support radius (R1,R2,R3):\n";
        std::cout<<"f(x,y,z,R1,R2,R3) = g(x,R1)*g(y,R2)*g(z,R3) (0<=|x|<=R1, 0<=|y|<=R2, 0<=|z|<=R3)\n";
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    std::cout<<"g(x,R) = (1/R)*(1-|x|/R) (0<=|x|<=R)\n";
}

//explicit instantiations
template class GridLinearWeightFunction<float,2>;
template class GridLinearWeightFunction<double,2>;
template class GridLinearWeightFunction<float,3>;
template class GridLinearWeightFunction<double,3>;

}  //end of namespace Physika
