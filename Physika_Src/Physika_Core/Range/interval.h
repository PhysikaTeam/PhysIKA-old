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

#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/type_utilities.h"

namespace Physika{

/*
 * Interval class is defined for C++ fundamental integer types and floating-point types.
 */

template <typename Scalar>
class Interval
{
public:
    Interval();
    explicit Interval(Scalar val);
    Interval(Scalar min_val, Scalar max_val);
    Interval(const Interval<Scalar> &interval);
    Interval<Scalar>& operator= (const Interval<Scalar> &interval);
    bool operator== (const Interval<Scalar> &interval);
    bool operator!= (const Interval<Scalar> &interval);
    ~Interval();
    Scalar center() const;
    Scalar size() const;
    Scalar minVal() const;
    Scalar maxVal() const;
    void setMinVal(Scalar val); //user is obligated to maintain validity of the interval
    void setMaxVal(Scalar val); //user is obligated to maintain validity of the interval
    bool inside(Scalar val) const;
    bool outside(Scalar val) const;

    static Interval<Scalar> unitInterval(); //[0,1]
protected:
    Scalar min_val_, max_val_;
private:
    void compileTimeCheck()
    {
        PHYSIKA_STATIC_ASSERT((is_integer<Scalar>::value||is_floating_point<Scalar>::value),
                              "Interval<Scalar> are only defined for integer types and floating-point types.");
    }
};

}  //end of namespace Physika

#endif  //PHYSIKA_CORE_RANGE_INTERVAL_H_
