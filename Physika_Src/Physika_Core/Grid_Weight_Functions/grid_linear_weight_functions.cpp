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
Scalar GridLinearWeightFunction<Scalar,Dim>::weight(const Vector<Scalar,Dim> &center_to_x) const
{
    LinearWeightFunction<Scalar,1> linear_1d;
    Scalar support_radius = 1.0; //support_radius is 1 grid cell
    Scalar scale = support_radius; //scale factor to enforce partition of unity (R)
    Scalar result = 1.0;
    for(unsigned int i = 0; i < Dim; ++i)
        result *= scale*linear_1d.weight(center_to_x[i],support_radius);
    return result;
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> GridLinearWeightFunction<Scalar,Dim>::gradient(const Vector<Scalar,Dim> &center_to_x) const
{
    LinearWeightFunction<Scalar,1> linear_1d;
    Scalar support_radius = 1.0; //support_radius is 1 grid cell
    Scalar scale = support_radius; //scale factor to enforce partition of unity (R)
    Vector<Scalar,Dim> result(1.0);
    for(unsigned int i = 0; i < Dim; ++i)
        for(unsigned int j = 0; j < Dim; ++j)
        {
            if(j==i)
                result[i] *= scale*linear_1d.gradient(center_to_x[j],support_radius);
            else
                result[i] *= scale*linear_1d.weight(center_to_x[j],support_radius);
        }
    return result;
}

template <typename Scalar, int Dim>
void GridLinearWeightFunction<Scalar,Dim>::printInfo() const
{
    switch(Dim)
    {
    case 2:
        std::cout<<"Grid-based linear weight function with support radius of 1 grid cell:\n";
        std::cout<<"f(x,y) = g(x)*g(y) (0<=|x|<=1, 0<=|y|<=1)\n";
        break;
    case 3:
        std::cout<<"Grid-based linear weight function with support radius of 1 grid cell:\n";
        std::cout<<"f(x,y,z) = g(x)*g(y)*g(z) (0<=|x|<=1, 0<=|y|<=1, 0<=|z|<=1)\n";
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    std::cout<<"g(x,R) = (1-|x|) (0<=|x|<=1)\n";
}

template <typename Scalar, int Dim>
Scalar GridLinearWeightFunction<Scalar,Dim>::supportRadius() const
{
    return 1.0;
}

//explicit instantiations
template class GridLinearWeightFunction<float,2>;
template class GridLinearWeightFunction<double,2>;
template class GridLinearWeightFunction<float,3>;
template class GridLinearWeightFunction<double,3>;

}  //end of namespace Physika
