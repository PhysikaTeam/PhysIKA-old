/*
 * @file interval.cpp
 * @brief 1D interval class [min,max]. 
 * @author FeiZhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */
#include "Interval.h"
#include "Utility/SimpleMath.h"

namespace PhysIKA {

template <typename Real>
COMM_FUNC Interval<Real>::Interval()
    : v0(0), v1(0), leftOpen(true), rightOpen(true)
{
}

template <typename Real>
COMM_FUNC Interval<Real>::Interval(Real min_val, Real max_val, bool lOpen, bool rOpen)
    : v0(min_val), v1(max_val), leftOpen(lOpen), rightOpen(rOpen)
{
}

template <typename Real>
COMM_FUNC Interval<Real>::Interval(const Interval<Real>& interval)
    : v0(interval.v0), v1(interval.v1), leftOpen(interval.leftOpen), rightOpen(interval.rightOpen)

{
}

template <typename Real>
COMM_FUNC Interval<Real>& Interval<Real>::operator=(const Interval<Real>& interval)
{
    v0        = interval.v0;
    v1        = interval.v1;
    leftOpen  = interval.leftOpen;
    rightOpen = interval.rightOpen;
    return *this;
}

template <typename Real>
COMM_FUNC bool Interval<Real>::operator==(const Interval<Real>& interval)
{
    return (v0 == interval.v0)
           && (v1 == interval.v0)
           && (leftOpen == interval.leftOpen)
           && (rightOpen == interval.rightOpen);
}

template <typename Real>
COMM_FUNC bool Interval<Real>::operator!=(const Interval<Real>& interval)
{
    return !((*this) == interval);
}

template <typename Real>
COMM_FUNC Interval<Real>::~Interval()
{
}

template <typename Real>
COMM_FUNC Real Interval<Real>::size() const
{
    return (v1 - v0);
}

template <typename Real>
inline COMM_FUNC Real Interval<Real>::leftLimit() const
{
    return v0;
}

template <typename Real>
inline COMM_FUNC Real Interval<Real>::rightLimit() const
{
    return v1;
}

template <typename Real>
COMM_FUNC bool Interval<Real>::isLeftOpen() const
{
    return leftOpen;
}

template <typename Real>
COMM_FUNC bool Interval<Real>::isRightOpen() const
{
    return rightOpen;
}

template <typename Real>
COMM_FUNC void Interval<Real>::setLeftLimit(Real val, bool bOpen)
{
    v0       = val;
    leftOpen = bOpen;
}

template <typename Real>
COMM_FUNC void Interval<Real>::setRightLimit(Real val, bool bOpen)
{
    v1        = val;
    rightOpen = bOpen;
}

template <typename Real>
COMM_FUNC bool Interval<Real>::inside(Real val) const
{
    if (isEmpty())
    {
        return false;
    }

    if (val > v0 && val < v1)
        return true;
    else if ((val == v0 && leftOpen == false) || (val == v1 && rightOpen == false))
        return true;
    else
        return false;
}

template <typename Real>
COMM_FUNC bool Interval<Real>::outside(Real val) const
{
    return !inside(val);
}

template <typename Real>
COMM_FUNC Interval<Real> Interval<Real>::intersect(const Interval<Real>& itv) const
{
    Interval<Real> ret;

    ret.v0        = max(v0, itv.v0);
    ret.v1        = min(v1, itv.v1);
    ret.leftOpen  = outside(ret.v0) || itv.outside(ret.v0);
    ret.rightOpen = outside(ret.v1) || itv.outside(ret.v1);

    return ret;
}

template <typename Real>
COMM_FUNC bool Interval<Real>::isEmpty() const
{
    return v0 > v1 || (v0 == v1 && (leftOpen == true || rightOpen == true));
}

template <typename Real>
COMM_FUNC Interval<Real> Interval<Real>::unitInterval()
{
    return Interval<Real>(0, 1, false, false);
}

}  //end of namespace PhysIKA
