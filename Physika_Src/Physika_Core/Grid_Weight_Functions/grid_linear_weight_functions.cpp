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

#include "Physika_Core/Weight_Functions/linear_weight_functions.h"
#include "Physika_Core/Grid_Weight_Functions/grid_linear_weight_functions.h"

namespace Physika{

template <typename Scalar, int Dim>
Scalar GridLinearWeightFunction<Scalar,Dim>::weight(const Vector<Scalar,Dim> &center_to_x, const Vector<Scalar,Dim> &support_radius) const
{
//TO DO
    return 0;
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> GridLinearWeightFunction<Scalar,Dim>::gradient(const Vector<Scalar,Dim> &center_to_x, const Vector<Scalar,Dim> &support_radius) const
{
//TO DO
    return Vector<Scalar,Dim>(0);
}

template <typename Scalar, int Dim>
void GridLinearWeightFunction<Scalar,Dim>::printInfo() const
{
//TO DO
}

}  //end of namespace Physika
