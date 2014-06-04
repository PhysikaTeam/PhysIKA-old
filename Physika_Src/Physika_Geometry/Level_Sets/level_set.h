/*
 * @file level_set.h 
 * @brief level set defined on uniform grid
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

#ifndef PHYSIKA_GEOMETRY_LEVEL_SETS_LEVEL_SET_H_
#define PHYSIKA_GEOMETRY_LEVEL_SETS_LEVEL_SET_H_

#include "Physika_Geometry/Cartesian_Grids/grid.h"
#include "Physika_Core/Arrays/array_Nd.h"

namespace Physika{

template <typename Scalar,int Dim>
class LevelSet
{
public:
protected:
    Grid<Scalar,Dim> grid_;
    ArrayND<Scalar,Dim> phi_;  //the level set value
};

}  //end of namespace Physika

#endif PHYSIKA_GEOMETRY_LEVEL_SETS_LEVEL_SET_H_
