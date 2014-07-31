/*
 * @file grid_cubic_weight_functions.h 
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

#ifndef PHYSIKA_CORE_GRID_WEIGHT_FUNCTIONS_GRID_CUBIC_WEIGHT_FUNCTIONS_H_
#define PHYSIKA_CORE_GRID_WEIGHT_FUNCTIONS_GRID_CUBIC_WEIGHT_FUNCTIONS_H_

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Grid_Weight_Functions/grid_weight_function.h"

namespace Physika{

/*
 * GridPiecewiseCubicSpline: dyadic product of 1D piece-wise cubic spline
 */

template <typename Scalar, int Dim>
class GridPiecewiseCubicSpline: public GridWeightFunction<Scalar,Dim>
{
public:
    GridPiecewiseCubicSpline(){}
    ~GridPiecewiseCubicSpline(){}
    Scalar weight(const Vector<Scalar,Dim> &center_to_x, const Vector<Scalar,Dim> &support_radius) const;
    Vector<Scalar,Dim> gradient(const Vector<Scalar,Dim> &center_to_x, const Vector<Scalar,Dim> &support_radius) const;
    void printInfo() const;
};

} //end of namespace Physika

#endif  //PHYSIKA_CORE_GRID_WEIGHT_FUNCTIONS_GRID_CUBIC_WEIGHT_FUNCTIONS_H_
