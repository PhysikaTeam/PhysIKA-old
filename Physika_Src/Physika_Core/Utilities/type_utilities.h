/*
 * @file type_utilities.h 
 * @brief utilities(methods&&structs&&macros) related to data types
 *        e.g.: check if two types are the same at compile time
 *        Latest c++ standards support type traits, we define our own
 *        utilities here such that Physika runs correctly on older compilers.
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

//is_folating_point: test if the type is a floating point type
template <typename T> 
struct is_floating_point
{ 
    static const bool value = (is_same<T,float>::value)||(is_same<T,double>::value)||(is_same<T,long double>::value);
};

//is_signed_integer: test if the type is a signed integer type
//signed char, signed short, signed int, signed long, signed long long
template <typename T>
struct is_signed_integer
{
    static const bool value = (is_same<T,signed char>::value)||(is_same<T,signed short>::value)||(is_same<T,signed int>::value)
        ||(is_same<T,signed long>::value)||(is_same<T,signed long long>::value);
};

//is_unsigned_integer: test if the type is a unsigned integer type
//unsigned char, unsigned short, unsigned int, unsigned long, unsigned long long
template <typename T>
struct is_unsigned_integer
{
    static const bool value = (is_same<T,unsigned char>::value)||(is_same<T,unsigned short>::value)||(is_same<T,unsigned int>::value)
        ||(is_same<T,unsigned long>::value)||(is_same<T,unsigned long long>::value);
};

//is_integer: test if the type is an integer type, signed or unsigned
template <typename T>
struct is_integer
{
    static const bool value = (is_unsigned_integer<T>::value)||(is_signed_integer<T>::value);
};

}  //end of namespace Physika

#endif //PHYSIKA_CORE_UTILITIES_TYPE_UTILITIES_H_
