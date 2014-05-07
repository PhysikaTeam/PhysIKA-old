/*
 * @file interval.h
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

#ifndef PHYSIKA_CORE_RANGE_INTERVAL_H_
#define PHYSIKA_CORE_RANGE_INTERVAL_H_

namespace Physika{

template <typename Scalar>
class Interval
{
public:
    Interval();
    explicit Interval(Scalar val);
    Interval(Scalar min_val, Scalar max_val);
    ~Interval();
    Scalar center() const;
    Scalar size() const;
    Scalar minVal() const;
    Scalar maxVal() const;
    bool inside(Scalar val) const;
    bool outside(Scalar val) const;
protected:
    Scalar min_val_, max_val_;
};

}  //end of namespace Physika

#endif  //PHYSIKA_CORE_RANGE_INTERVAL_H_










