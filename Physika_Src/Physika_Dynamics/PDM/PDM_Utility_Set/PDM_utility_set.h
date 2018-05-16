/*
 * @file PDM_utility_set.h 
 * @brief third-party utility functions which would be used in different classes  
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


#ifndef PHYSIKA_DYNAMICS_PDM_PDM_UTILITY_SET_PDM_UTILITY_SET_H
#define PHYSIKA_DYNAMICS_PDM_PDM_UTILITY_SET_PDM_UTILITY_SET_H

#include <vector>

namespace Physika{

template <typename Scalar, int Dim> class Vector;

///////////////////////////////////////////Global & Inline function/////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////
template <typename Scalar, int Dim>
class PDMUtilitySet
{
public:
    //compute min and max corner given a set of vertex
    static Vector<Scalar, Dim> computeMinCorner(const std::vector<Vector<Scalar, Dim> > & ele_pos_vec);
    static Vector<Scalar, Dim> computeMaxCorner(const std::vector<Vector<Scalar, Dim> > & ele_pos_vec);
};

}//end of namespace Physika

#endif //PHYSIKA_DYNAMICS_PDM_PDM_UTILITY_SET_PDM_UTILITY_SET_H