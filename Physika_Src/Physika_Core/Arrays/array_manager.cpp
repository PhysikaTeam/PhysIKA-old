/*
 * @file array_manager.cpp 
 * @brief array class. To reorder the elements of 1D arrays concurrently.
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

#include "Physika_Core/Arrays/array_manager.h"

namespace Physika{

void ArrayManager::addArray(std::string key, ReorderObject *arr)
{
    arrays_.insert(std::map<std::string, ReorderObject*>::value_type(key, arr));
}


ReorderObject* ArrayManager::getArray(std::string key)
{
    return arrays_[key];
}

void ArrayManager::reorder(unsigned int* ids, unsigned int size)
{
    std::map<std::string, ReorderObject*>::iterator iter;
    for (iter = arrays_.begin(); iter != arrays_.end(); ++iter)
    {
	std::cout<<iter->first<<" ";
	iter->second->reorder(ids, size);
    }
}

}  //end of namespace Physika















