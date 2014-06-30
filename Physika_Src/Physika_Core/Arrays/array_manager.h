/*
 * @file array_manager.h 
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

#ifndef PHSYIKA_CORE_ARRAYS_ARRAY_MANAGER_H_
#define PHSYIKA_CORE_ARRAYS_ARRAY_MANAGER_H_

#include <map>
#include <string>
#include "Physika_Core/Arrays/array.h"

namespace Physika{

class ArrayManager
{
public:
    ArrayManager(){}
    ~ArrayManager(){}

    void addArray(std::string key, ArrayBase *arr);

    ArrayBase* getArray(std::string key);

    void permutate(unsigned int* ids, unsigned int size);

private:
    std::map<std::string, ArrayBase*> arrays_;
};

}  //end of namespace Physika

#endif //PHSYIKA_CORE_ARRAYS_ARRAY_MANAGER_H_










