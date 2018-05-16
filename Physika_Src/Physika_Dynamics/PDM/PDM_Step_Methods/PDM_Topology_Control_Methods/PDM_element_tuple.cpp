/*
 * @file PDM_element_tuple.cpp 
 * @brief class PDMElementTuple.
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

#include <algorithm>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Topology_Control_Methods/PDM_element_tuple.h"

namespace Physika{

PDMElementTuple::PDMElementTuple()
{

}

PDMElementTuple::~PDMElementTuple()
{

}

void PDMElementTuple::setElementVec(unsigned int fir_ele, unsigned int sec_ele)
{
    this->ele_vec_.push_back(fir_ele);
    this->ele_vec_.push_back(sec_ele);
    std::sort(this->ele_vec_.begin(), this->ele_vec_.end());
}

const std::vector<unsigned int> & PDMElementTuple::eleVec() const
{
    return this->ele_vec_;
}

bool PDMElementTuple::operator<(const PDMElementTuple & rhs_ele_tuple) const
{
    PHYSIKA_ASSERT(this->ele_vec_.size()==2 && rhs_ele_tuple.ele_vec_.size()==2);
    for (unsigned int i=0; i<2; i++)
    {
        if (this->ele_vec_[i]<rhs_ele_tuple.ele_vec_[i])
            return true;
        if (this->ele_vec_[i]>rhs_ele_tuple.ele_vec_[i])
            return false;
    }
    return false;
}

}//end of namespace Physika