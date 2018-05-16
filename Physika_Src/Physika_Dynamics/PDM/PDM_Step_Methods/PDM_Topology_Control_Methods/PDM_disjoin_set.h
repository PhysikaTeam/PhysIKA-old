/*
 * @file PDM_disjoin_set.h 
 * @brief class PDMDisjonSet.
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

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_TOPOLOGY_CONTROL_METHODS_PDM_DISJOIN_SET_H
#define PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_TOPOLOGY_CONTROL_METHODS_PDM_DISJOIN_SET_H

#include <set>
#include <vector>
#include <map>

namespace Physika{

class PDMDisjoinSet
{
public:
    PDMDisjoinSet();
    PDMDisjoinSet(const std::set<unsigned int> & data_set);
    PDMDisjoinSet(const std::vector<unsigned int> & data_set);
    void makeSet(const std::set<unsigned int> & data_set);
    void makeSet(const std::vector<unsigned int> & data_set);
    unsigned int findSet(unsigned int x);
    void unionSet(unsigned int x, unsigned int y);

protected:
    std::map<unsigned int, unsigned int> parent_map_;

};

}//end of namespace Physika

#endif //PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_TOPOLOGY_CONTROL_METHODS_PDM_DISJOIN_SET_H