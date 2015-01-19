/*
 * @file math_utilities.h 
 * @brief This file is used to define math constants and functions frequently used in Physika.
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

#ifndef PHYSIKA_CORE_UTILITIES_MATH_UTILITIES_H_
#define PHYSIKA_CORE_UTILITIES_MATH_UTILITIES_H_

#include <limits>
#include <cmath>

namespace Physika{

////////////////////////////////constants//////////////////////////////////////////////
const double PI = 3.14159265358979323846;
const double E = 2.71828182845904523536;
const float FLOAT_EPSILON = std::numeric_limits<float>::epsilon();
const double DOUBLE_EPSILON = std::numeric_limits<double>::epsilon();

///////////////////////////////functions/////////////////////////////////////////////////
/*
 * Function List: Please update the list everytime you add/remove a function!!!
 * abs(); sqrt(); max(); min(); isEqual();
 */

/*
 * abs(), sqrt() are replacement for functions from std because some compilers do not
 * support sqrt and abs of integer type
 */
template <typename Scalar>
inline Scalar abs(Scalar value)
{
    return value>=0?value:-value;
}

inline float sqrt(float value)
{
    return std::sqrt(value);
}

inline double sqrt(double value)
{
    return std::sqrt(value);
}

inline long double sqrt(long double value)
{
    return std::sqrt(value);
}

template <typename Scalar>
inline double sqrt(Scalar value)
{
    return std::sqrt(static_cast<double>(value));
}

#undef max //undefine the max in WinDef.h
template <typename Scalar>
inline Scalar max(Scalar lhs, Scalar rhs)
{
	return lhs > rhs ? lhs : rhs;	
}

#undef min //undefine the min in WinDef.h
template <typename Scalar>
inline Scalar min(Scalar lhs, Scalar rhs)
{
	return lhs < rhs ? lhs : rhs;
}

//compare if two floating point numbers are equal
//ref: http://floating-point-gui.de/errors/comparison/
template <typename Scalar>
inline bool isEqual(Scalar a, Scalar b, Scalar relative_tolerance = 1.0e-6)
{
    Scalar abs_a = abs(a), abs_b = abs(b), diff = abs(a-b);
    Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
    if(a == b)
        return true;
    else if(a==0||b==0||diff<epsilon)  //absolute tolerance for near zero values
        return diff < epsilon;
    else  //relative tolerance for others
        return diff/(abs_a+abs_b) < relative_tolerance;
}

}  //end of namespace Physika

#endif //PHYSIKA_CORE_UTILITIES_MATH_UTILITIES_H_
