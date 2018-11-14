/*
 * @file PDM_utility-set.cpp
 * @brief Utility for PDM drivers.
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Dynamics/PDM/PDM_Utility_Set/PDM_utility_set.h"

namespace Physika{

template <typename Scalar, int Dim>
Vector<Scalar, Dim> PDMUtilitySet<Scalar, Dim>::computeMinCorner(const std::vector<Vector<Scalar, Dim> > & ele_pos_vec)
{
    PHYSIKA_ASSERT(ele_pos_vec.size() > 0);
    Vector<Scalar, Dim> min_corner = ele_pos_vec[0];
    for (unsigned int i = 1; i < ele_pos_vec.size(); i++)
    {
        for (unsigned int j = 0; j < Dim; j++)
            min_corner[j] = min(min_corner[j], ele_pos_vec[i][j]);
    }
    return min_corner;
}

template <typename Scalar, int Dim>
Vector<Scalar, Dim> PDMUtilitySet<Scalar, Dim>::computeMaxCorner(const std::vector<Vector<Scalar, Dim> > & ele_pos_vec)
{
    PHYSIKA_ASSERT(ele_pos_vec.size() > 0);
    Vector<Scalar, Dim> max_corner = ele_pos_vec[0];
    for (unsigned int i = 1; i < ele_pos_vec.size(); i++)
    {
        for (unsigned int j = 0; j < Dim; j++)
            max_corner[j] = max(max_corner[j], ele_pos_vec[i][j]);
    }
    return max_corner;
}

//explicit instantiations
template class PDMUtilitySet<float, 2>;
template class PDMUtilitySet<double,2>;
template class PDMUtilitySet<float, 3>;
template class PDMUtilitySet<double,3>;

}//end of namespace Physika