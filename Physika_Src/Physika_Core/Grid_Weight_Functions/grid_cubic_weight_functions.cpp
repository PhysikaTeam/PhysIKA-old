/*
 * @file grid_cubic_weight_functions.cpp 
 * @brief collection of cubic grid-based weight functions. 
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
#include "Physika_Core/Grid_Weight_Functions/grid_cubic_weight_functions.h"
#include "Physika_Core/Weight_Functions/cubic_weight_functions.h"

namespace Physika{

template <typename Scalar, int Dim>
Scalar GridPiecewiseCubicSpline<Scalar,Dim>::weight(const Vector<Scalar,Dim> &center_to_x) const
{
    PiecewiseCubicSpline<Scalar,1> cubic_spline_1d;
    Scalar support_radius = 2.0;  //support radius is 2 grid cells
    Scalar scale = 0.5*support_radius; //scale factor to enforce partition of unity (h = 0.5*R)
    Scalar result = 1.0;
    for(unsigned int i = 0; i < Dim; ++i)
        result *= scale*cubic_spline_1d.weight(center_to_x[i],support_radius);
    return result;
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> GridPiecewiseCubicSpline<Scalar,Dim>::gradient(const Vector<Scalar,Dim> &center_to_x) const
{
    PiecewiseCubicSpline<Scalar,1> cubic_spline_1d;
    Scalar support_radius = 2.0;  //support radius is 2 grid cells
    Scalar scale = 0.5*support_radius; //scale factor to enforce partition of unity (h = 0.5*R)
    Vector<Scalar,Dim> result(1.0);
    for(unsigned int i = 0; i < Dim; ++i)
        for(unsigned int j = 0; j < Dim; ++j)
        {
            if(j==i)
                result[i] *= scale*cubic_spline_1d.gradient(center_to_x[j],support_radius);
            else
                result[i] *= scale*cubic_spline_1d.weight(center_to_x[j],support_radius);
        }
    return result;
}

template <typename Scalar, int Dim>
void GridPiecewiseCubicSpline<Scalar,Dim>::printInfo() const
{
    switch(Dim)
    {
    case 2:
        std::cout<<"Grid-based piece-wise cubic spline with support radius of 2 cell size:\n";
        std::cout<<"f(x,y) = g(x)*g(y) (0<=|x|<=2, 0<=|y|<=2)\n";
        break;
    case 3:
        std::cout<<"Grid-based piece-wise cubic spline with support radius 2 cell size:\n";
        std::cout<<"f(x,y,z) = g(x)*g(y)*g(z) (0<=|x|<=2, 0<=|y|<=2, 0<=|z|<=2)\n";
        break;
    default:
        PHYSIKA_ERROR("Wrong dimension specified.");
    }
    std::cout<<"g(x) is 1D piecewise cubic spline with support radius R = 2: \n";
    std::cout<<"g(x) = (2/3-|x|^2+1/2*|x|^3) (0<=|x|<=1)\n";
    std::cout<<"g(x) = (2-|x|)^3/6 (1<=|x|<=2)\n";
}

template <typename Scalar, int Dim>
Scalar GridPiecewiseCubicSpline<Scalar,Dim>::supportRadius() const
{
    return 2.0;
}

//explicit instantiations
template class GridPiecewiseCubicSpline<float,2>;
template class GridPiecewiseCubicSpline<double,2>;
template class GridPiecewiseCubicSpline<float,3>;
template class GridPiecewiseCubicSpline<double,3>;

}  //end of namespace Physika
