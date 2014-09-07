/*
 * @file interval.cpp
 * @brief 1D interval class [min,max]. 
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

#include <cstdlib>
#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/type_utilities.h"
#include "Physika_Core/Range/interval.h"

namespace Physika{

template <typename Scalar>
Interval<Scalar>::Interval():min_val_(0),max_val_(0)
{
    PHYSIKA_STATIC_ASSERT((is_integer<Scalar>::value||is_floating_point<Scalar>::value),
                      "Interval<Scalar> are only defined for integer types and floating-point types.");
}

template <typename Scalar>
Interval<Scalar>::Interval(Scalar val):min_val_(val),max_val_(val)
{
    PHYSIKA_STATIC_ASSERT((is_integer<Scalar>::value||is_floating_point<Scalar>::value),
                      "Interval<Scalar> are only defined for integer types and floating-point types.");
}

template <typename Scalar>
Interval<Scalar>::Interval(Scalar min_val, Scalar max_val)
{
    PHYSIKA_STATIC_ASSERT((is_integer<Scalar>::value||is_floating_point<Scalar>::value),
                      "Interval<Scalar> are only defined for integer types and floating-point types.");
    if(min_val>max_val)
    {
        std::cerr<<"Minimum value of interval must be equal or smaller than maximum value!\n";
        std::exit(EXIT_FAILURE);
    }
    min_val_ = min_val;
    max_val_ = max_val;
}

template <typename Scalar>
Interval<Scalar>::Interval(const Interval<Scalar> &interval)
    :min_val_(interval.min_val_),max_val_(interval.max_val_)
{
}

template <typename Scalar>
Interval<Scalar>& Interval<Scalar>::operator= (const Interval<Scalar> &interval)
{
    min_val_ = interval.min_val_;
    max_val_ = interval.max_val_;
    return *this;
}

template <typename Scalar>
bool Interval<Scalar>::operator==(const Interval<Scalar> &interval)
{
    return (min_val_==interval.min_val_)&&(max_val_==interval.max_val_);
}

template <typename Scalar>
bool Interval<Scalar>::operator!=(const Interval<Scalar> &interval)
{
    return !((*this)==interval);
}

template <typename Scalar>
Interval<Scalar>::~Interval()
{
}

template <typename Scalar>
Scalar Interval<Scalar>::center() const
{
    return (max_val_+min_val_)/2;
}

template <typename Scalar>
Scalar Interval<Scalar>::size() const
{
    return (max_val_-min_val_);
}

template <typename Scalar>
Scalar Interval<Scalar>::minVal() const
{
    return min_val_;
}

template <typename Scalar>
Scalar Interval<Scalar>::maxVal() const
{
    return max_val_;
}

template <typename Scalar>
void Interval<Scalar>::setMinVal(Scalar val)
{
    min_val_ = val;
}

template <typename Scalar>
void Interval<Scalar>::setMaxVal(Scalar val)
{
    max_val_ = val;
}

template <typename Scalar>
bool Interval<Scalar>::inside(Scalar val) const
{
    if(val>=min_val_&&val<=max_val_)
        return true;
    else
        return false;   
}

template <typename Scalar>
bool Interval<Scalar>::outside(Scalar val) const
{
    return !inside(val);
}

template <typename Scalar>
Interval<Scalar> Interval<Scalar>::unitInterval()
{
    return Interval<Scalar>(0,1);
}

//explicit instantiation
template class Interval<unsigned char>;
template class Interval<unsigned short>;
template class Interval<unsigned int>;
template class Interval<unsigned long>;
template class Interval<unsigned long long>;
template class Interval<signed char>;
template class Interval<short>;
template class Interval<int>;
template class Interval<long>;
template class Interval<long long>;
template class Interval<float>;
template class Interval<double>;
template class Interval<long double>;

} //end of namespace Physika
