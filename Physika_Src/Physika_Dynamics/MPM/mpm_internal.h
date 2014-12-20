/*
 * @file mpm_internal.h 
 * @Brief data structures internally used by MPM drivers.
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

#ifndef PHYSIKA_DYNAMICS_MPM_MPM_INTERNAL_H_
#define PHYSIKA_DYNAMICS_MPM_MPM_INTERNAL_H_

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

namespace MPMInternal{

/*
 * NodeIndexWeightGradientPair:
 * pair of the grid node and its interpolation weight/gradient value
 * for uniform grid
 */

template <typename Scalar,int Dim>
struct NodeIndexWeightGradientPair
{
    Vector<unsigned int,Dim> node_idx_;
    Scalar weight_value_;
    Vector<Scalar,Dim> gradient_value_;
};

/*
 * NodeIndexWeightPair:
 * pair of the grid node and its interpolation weight value
 * for uniform grid
 */

template <typename Scalar,int Dim>
struct NodeIndexWeightPair
{
    Vector<unsigned int,Dim> node_idx_;
    Scalar weight_value_;
};

}  //end of namespace MPMInternal

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_MPM_INTERNAL_H_
