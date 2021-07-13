/*
 * @file math_utilities.h
 * @brief This file is used to define math constants and functions frequently used in PhysIKA.
 * @author FeiZhu
 *
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
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
#include "Core/Platform.h"

namespace PhysIKA {

////////////////////////////////constants//////////////////////////////////////////////

//Note: use "constexpr" instead "const" for cuda codes

constexpr double PI             = 3.14159265358979323846;
constexpr double E              = 2.71828182845904523536;
constexpr float  FLOAT_EPSILON  = std::numeric_limits<float>::epsilon();
constexpr double DOUBLE_EPSILON = std::numeric_limits<double>::epsilon();
constexpr float  FLOAT_MAX      = (std::numeric_limits<float>::max)();
constexpr double DOUBLE_MAX     = (std::numeric_limits<double>::max)();
constexpr float  FLOAT_MIN      = std::numeric_limits<float>::lowest();
constexpr double DOUBLE_MIN     = std::numeric_limits<double>::lowest();

#define M_PI 3.14159265358979323846
#define M_E 2.71828182845904523536

///////////////////////////////functions/////////////////////////////////////////////////
/*
 * Function List: Please update the list every time you add/remove a function!!!
 * abs(); sqrt(); cbrt(); max(); min(); isEqual();
 */

/*
 * abs(), sqrt() are replacement for functions from std because some compilers do not
 * support sqrt and abs of integer type
 */
template <typename Scalar>
COMM_FUNC Scalar abs(Scalar value)
{
    return value >= 0 ? value : -value;
}

// inline float sqrt(float value)
// {
//     return std::sqrt(value);
// }
//
// inline double sqrt(double value)
// {
//     return std::sqrt(value);
// }

// inline long double sqrt(long double value)
// {
//     return std::sqrt(value);
// }

// template <typename Scalar>
// inline double sqrt(Scalar value)
// {
//     return std::sqrt(static_cast<double>(value));
// }

inline float cbrt(float value)
{
    float base = value > 0.0f ? value : -value;
    float sign = value > 0.0f ? 1.0f : -1.0f;
    return sign * std::pow(base, 1.0f / 3.0f);
}

inline double cbrt(double value)
{
    double base = value > 0.0 ? value : -value;
    double sign = value > 0.0 ? 1.0 : -1.0;
    return sign * std::pow(base, 1.0 / 3.0);
}

inline long double cbrt(long double value)
{
    long double base = value > 0.0 ? value : -value;
    long double sign = value > 0.0 ? 1.0 : -1.0;
    return sign * std::pow(base, static_cast<long double>(1.0 / 3.0));
}

template <typename Scalar>
inline double cbrt(Scalar value)
{
    return cbrt(static_cast<double>(value));
}

#undef max  //undefine the max in WinDef.h
template <typename Scalar>
inline Scalar max(Scalar lhs, Scalar rhs)
{
    return lhs > rhs ? lhs : rhs;
}

// #undef min //undefine the min in WinDef.h
// template <typename Scalar>
// inline Scalar min(Scalar lhs, Scalar rhs)
// {
//     return lhs < rhs ? lhs : rhs;
// }

//compare if two floating point numbers are equal
//ref: http://floating-point-gui.de/errors/comparison/
template <typename Scalar>
COMM_FUNC bool isEqual(Scalar a, Scalar b, double relative_tolerance = 1.0e-6)
{
    Scalar abs_a = abs(a), abs_b = abs(b), diff = abs(a - b);
    Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
    if (a == b)
        return true;
    else if (a == 0 || b == 0 || diff < epsilon)  //absolute tolerance for near zero values
        return diff < epsilon;
    else  //relative tolerance for others
        return diff / (abs_a + abs_b) < relative_tolerance;
}

}  //end of namespace PhysIKA

#endif  //PHYSIKA_CORE_UTILITIES_MATH_UTILITIES_H_
