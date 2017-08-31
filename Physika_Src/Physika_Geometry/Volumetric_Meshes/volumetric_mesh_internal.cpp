/*
 * @file  volumetric_mesh_internal.cpp
 * @brief Internal types for VolumetricMesh
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <iostream>
#include <algorithm>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh_internal.h"
using std::string;
using std::vector;

namespace Physika{

namespace VolumetricMeshInternal{

Region::Region()
{
}

Region::Region(const string &region_name, const vector<unsigned int> &elements)
    :name_(region_name),elements_(elements)
{
}

Region::Region(const Region &region)
{
    this->name_ = region.name_;
    this->elements_ = region.elements_;
}

Region::~Region()
{
}

Region& Region::operator= (const Region &region)
{
    this->name_ = region.name_;
    this->elements_ = region.elements_;
    return *this;
}

const string& Region::name() const
{
    return name_;
}

void Region::setName(const string &new_name)
{
    name_ = new_name;
}

unsigned int Region::elementNum() const
{
    return elements_.size();
}

const vector<unsigned int>& Region::elements() const
{
    return elements_;
}

void Region::addElement(unsigned int new_ele_idx)
{
    elements_.push_back(new_ele_idx);
}

void Region::removeElement(unsigned int ele_idx)
{
    int local_idx = this->elementLocalIndex(ele_idx);
    if(local_idx>=0)
        removeElementAtIndex(local_idx);
    else
        std::cerr<<"Element "<<ele_idx<<" not in this region!\n";
}

void Region::removeElementAtIndex(unsigned int ele_idx_in_region)
{
    PHYSIKA_ASSERT(ele_idx_in_region<elements_.size());
    vector<unsigned int>::iterator iter = elements_.begin() + ele_idx_in_region;
    elements_.erase(iter);
}

int Region::elementLocalIndex(unsigned int ele_idx)
{
    vector<unsigned int>::iterator iter = find(elements_.begin(),elements_.end(),ele_idx);
    if(iter==elements_.end())
        return -1;
    else
        return static_cast<int>(iter-elements_.begin());
}

}  //end of namespace VolumetricMeshInternal

} //end of namespace Physika
