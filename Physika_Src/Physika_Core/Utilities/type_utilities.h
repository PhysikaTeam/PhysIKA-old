/*
 * @file type_utilities.h 
 * @brief utilities(methods&&structs&&macros) related to data types
 *        e.g.: check if two types are the same at compile time
 * @author FeiZhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_UTILITIES_TYPE_UTILITIES_H_
#define PHYSIKA_CORE_UTILITIES_TYPE_UTILITIES_H_

namespace Physika{

//is_same: if two types are the same, value is true; otherwise, false
//the name 'is_same' violates Physika naming rule in order to agree with std::is_same (C++11)
template <typename T1, typename T2> struct is_same { static const bool value = false; };
template <typename T> struct is_same<T,T> { static const bool value = true; };

}  //end of namespace Physika

#endif //PHYSIKA_CORE_UTILITIES_TYPE_UTILITIES_H_
















