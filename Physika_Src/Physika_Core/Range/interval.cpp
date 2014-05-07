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

#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Range/interval.h"

namespace Physika{

template <typename Scalar>
Interval<Scalar>::Interval():min_val_(0),max_val_(0)
{
}

template <typename Scalar>
Interval<Scalar>::Interval(Scalar val):min_val_(val),max_val_(val)
{
}

template <typename Scalar>
Interval<Scalar>::Interval(Scalar min_val, Scalar max_val)
{
    PHYSIKA_ASSERT(min_val<=max_val);
    min_val_ = min_val;
    max_val_ = max_val;
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

//explicit instantiation
template class Interval<float>;
template class Interval<double>;

} //end of namespace Physika


















