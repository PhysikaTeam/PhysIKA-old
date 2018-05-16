/*
 * @file PDM_element_tuple.h 
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


#ifndef PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_TOPOLOGY_CONTROL_METHODS_PDM_ELEMENT_TUPLE_H
#define PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_TOPOLOGY_CONTROL_METHODS_PDM_ELEMENT_TUPLE_H

#include <vector>

namespace Physika{

class PDMElementTuple
{
public:
    PDMElementTuple();
    ~PDMElementTuple();

    void setElementVec(unsigned int fir_ele, unsigned int sec_ele);
    const std::vector<unsigned int> & eleVec() const;

    bool operator < (const PDMElementTuple & rhs_ele_tuple) const;
    
protected:
    std::vector<unsigned int> ele_vec_;
};

}//end of namespace Physika
#endif //PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_TOPOLOGY_CONTROL_METHODS_PDM_ELEMENT_TUPLE_H 