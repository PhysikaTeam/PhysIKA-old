/*
 * @file PDM_disjoin_set.cpp
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

#include <iostream>
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Topology_Control_Methods/PDM_disjoin_set.h"

namespace Physika{

PDMDisjoinSet::PDMDisjoinSet()
{

}

PDMDisjoinSet::PDMDisjoinSet(const std::set<unsigned int> & data_set)
{
    this->makeSet(data_set);
}

PDMDisjoinSet::PDMDisjoinSet(const std::vector<unsigned int> & data_set)
{
    this->makeSet(data_set);
}

void PDMDisjoinSet::makeSet(const std::set<unsigned int> & data_set)
{
    this->parent_map_.clear();
    for (std::set<unsigned int>::const_iterator iter = data_set.begin(); iter != data_set.end(); iter++)
    {
        this->parent_map_[*iter] = *iter;
    }
}

void PDMDisjoinSet::makeSet(const std::vector<unsigned int> & data_set)
{
    this->parent_map_.clear();
    for (unsigned int i=0; i<data_set.size(); i++)
    {
        this->parent_map_[data_set[i]] = data_set[i];
    }
}

unsigned int PDMDisjoinSet::findSet(unsigned int x)
{
    std::map<unsigned int, unsigned int>::iterator iter = this->parent_map_.find(x);
    if (iter == this->parent_map_.end())
    {
        std::cerr<<"error: element is not find in disjoin set!\n";
        std::exit(EXIT_FAILURE);
    }
    if (iter->first != iter->second)
    {
        iter->second = this->findSet(iter->second);
    }
    return iter->second;
}

void PDMDisjoinSet::unionSet(unsigned int x, unsigned int y)
{
    unsigned int p_x = this->findSet(x);
    unsigned int p_y = this->findSet(y);
    if (p_x != p_y)
    {
        std::map<unsigned int, unsigned int>::iterator px_iter = this->parent_map_.find(p_x);
        px_iter->second = p_y;
    }
}


}// end of namespace Physika