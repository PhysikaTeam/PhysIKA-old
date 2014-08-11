/*
 * @file grid_linear_weight_functions.h 
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

#ifndef PHYSIKA_CORE_GRID_WEIGHT_FUNCTIONS_GRID_LINEAR_WEIGHT_FUNCTIONS_H_
#define PHYSIKA_CORE_GRID_WEIGHT_FUNCTIONS_GRID_LINEAR_WEIGHT_FUNCTIONS_H_

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Grid_Weight_Functions/grid_weight_function.h"

namespace Physika{

/*
 * GridLinearWeightFunction: dyadic product of the most common 1D linear weight function.
 * Support radius: 1 grid cell.
 */

template <typename Scalar, int Dim>
class GridLinearWeightFunction: public GridWeightFunction<Scalar,Dim>
{
public:
    GridLinearWeightFunction(){}
    ~GridLinearWeightFunction(){}
    Scalar weight(const Vector<Scalar,Dim> &center_to_x) const;
    Vector<Scalar,Dim> gradient(const Vector<Scalar,Dim> &center_to_x) const;
    void printInfo() const;
    Scalar supportRadius() const;
};

}  //end of namespace Physika

#endif //PHYSIKA_CORE_GRID_WEIGHT_FUNCTIONS_GRID_LINEAR_WEIGHT_FUNCTIONS_H_
