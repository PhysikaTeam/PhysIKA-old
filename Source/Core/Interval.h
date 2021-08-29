/*
 * @file interval.h
 * @brief 1D interval class [min,max]. 
 * @author FeiZhu, Xiaowei He
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef INTERVAL_H
#define INTERVAL_H

#include "Core/Platform.h"

namespace PhysIKA {

/*
     * Interval class is defined for C++ floating-point types.
     */

template <typename Real>
class Interval
{
public:
    COMM_FUNC Interval();
    COMM_FUNC Interval(Real min_val, Real max_val, bool lOpen = false, bool rOpen = false);
    COMM_FUNC Interval(const Interval<Real>& interval);
    COMM_FUNC Interval<Real>& operator=(const Interval<Real>& interval);
    COMM_FUNC bool            operator==(const Interval<Real>& interval);
    COMM_FUNC bool            operator!=(const Interval<Real>& interval);
    COMM_FUNC ~Interval();

    COMM_FUNC Real size() const;

    inline COMM_FUNC Real leftLimit() const;
    inline COMM_FUNC Real rightLimit() const;

    COMM_FUNC bool isLeftOpen() const;
    COMM_FUNC bool isRightOpen() const;

    COMM_FUNC void setLeftLimit(Real val, bool bOpen = false);
    COMM_FUNC void setRightLimit(Real val, bool bOpen = false);

    COMM_FUNC bool inside(Real val) const;
    COMM_FUNC bool outside(Real val) const;

    COMM_FUNC Interval<Real> intersect(const Interval<Real>& itv) const;

    COMM_FUNC bool isEmpty() const;

    COMM_FUNC static Interval<Real> unitInterval();  //[0,1]
private:
    Real v0, v1;
    bool leftOpen, rightOpen;
};

template class Interval<float>;
template class Interval<double>;

}  //end of namespace PhysIKA

#endif  //PHYSIKA_CORE_RANGE_INTERVAL_H_
