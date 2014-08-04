/*
 * @file grid_weight_function.h 
 * @brief base class of all weight functions whose support domain is a grid, abstract class. 
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

#ifndef PHYSIKA_CORE_GRID_WEIGHT_FUNCTIONS_GRID_WEIGHT_FUNCTION_H_
#define PHYSIKA_CORE_GRID_WEIGHT_FUNCTIONS_GRID_WEIGHT_FUNCTION_H_

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

/*
 * GridWeightFunction: Base class of grid-based weight functions,
 * i.e., weight functions whose support domain is a rectangle in 2D
 * and a cuboid in 3D. It is constructed via dyadic product of weight functions
 * in 1D.
 *
 * These weight functions are used in methods that involve cartesian grids.
 *
 */

template <typename Scalar, int Dim>
class GridWeightFunction
{
public:
    GridWeightFunction(){}
    virtual ~GridWeightFunction(){}
    virtual Scalar weight(const Vector<Scalar,Dim> &center_to_x, const Vector<Scalar,Dim> &support_radius) const=0;
    virtual Vector<Scalar,Dim> gradient(const Vector<Scalar,Dim> &center_to_x, const Vector<Scalar,Dim> &support_radius) const=0;
    virtual void printInfo() const=0;

    typedef Scalar ScalarType;
    static const int DimSize = Dim;
};
}  //end of namespace Physika

#endif //PHYSIKA_CORE_GRID_WEIGHT_FUNCTIONS_GRID_WEIGHT_FUNCTION_H_
