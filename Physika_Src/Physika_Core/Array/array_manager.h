/*
 * @file array_manager.h 
 * @brief array class. To reorder the array;
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
#ifndef PHSYIKA_CORE_ARRAY_ARRAY_MANAGER_H_
#define PHSYIKA_CORE_ARRAY_ARRAY_MANAGER_H_


#include "Physika_Core/Array/array.h"
namespace Physika{

class ArrayManager
{
public:
    ArrayManager(){};
    ~ArrayManager(){};

    void addArray(std::string key, ReorderObject *arr)
    {
        arrs.insert(std::map<std::string, ReorderObject*>::value_type(key, arr));
    }

    ReorderObject* getArray(std::string key)
    {
        return arrs[key];
    }

    void reorder(unsigned int* ids, unsigned int size)
    {
        std::map<std::string, ReorderObject*>::iterator iter;

        for (iter = arrs.begin(); iter != arrs.end(); iter++)
        {
            std::cout<<iter->first<<" ";
            iter->second->reorder(ids, size);
        }
    }

private:
    std::map<std::string, ReorderObject*> arrs;
};

}
#endif //PHSYIKA_CORE_ARRAY_ARRAY_MANAGER_H_
