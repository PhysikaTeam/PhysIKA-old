/*
 * @file array_manager.cpp 
 * @brief array manager class, perform operations on the elements of several 1D arrays concurrently.
 * @author Sheng Yang, Fei Zhu
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

void ArrayManager::addArray(std::string key, ArrayBase *arr)
{
    arrays_.insert(std::map<std::string, ArrayBase*>::value_type(key, arr));
}


ArrayBase* ArrayManager::getArray(std::string key)
{
    return arrays_[key];
}

void ArrayManager::permutate(unsigned int* ids, unsigned int size)
{
    std::map<std::string, ArrayBase*>::iterator iter;
    for (iter = arrays_.begin(); iter != arrays_.end(); ++iter)
    {
        std::cout<<iter->first<<" ";
        iter->second->permutate(ids, size);
    }
}

}  //end of namespace Physika
