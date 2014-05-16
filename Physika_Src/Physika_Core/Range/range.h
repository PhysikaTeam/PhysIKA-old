/*
 * @file range.h
 * @brief higher dimensional counterpart of interval class.
 *        2D example: (1,1) to (3,3) 
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

#ifndef PHYSIKA_CORE_RANGE_RANGE_H_
#define PHYSIKA_CORE_RANGE_RANGE_H_

#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/type_utilities.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

template <typename Scalar,int Dim>
class Range
{
public:
    Range();
    explicit Range(const Vector<Scalar,Dim> &point);
    Range(const Vector<Scalar,Dim> &min_val, const Vector<Scalar,Dim> &max_val);
    Range(const Range<Scalar,Dim> &range);
    Range<Scalar,Dim>& operator= (const Range<Scalar,Dim> &range);
    bool operator== (const Range<Scalar,Dim> &range) const;
    ~Range();
    Vector<Scalar,Dim> center() const;
    Vector<Scalar,Dim> edgeLengths() const;
    Scalar size() const;  //2D: area; 3D: volume
    const Vector<Scalar,Dim>& minCorner() const;
    const Vector<Scalar,Dim>& maxCorner() const;
    bool inside(const Vector<Scalar,Dim> &val) const;
    bool outside(const Vector<Scalar,Dim> &val) const;

    static Range<Scalar,Dim> unitRange();  //each dimension is in range [0,1]
protected:
    Vector<Scalar,Dim> min_corner_, max_corner_;
protected:
    PHYSIKA_STATIC_ASSERT(Dim==3||Dim==2,"Range<Scalar,Dim> are only defined for Dim==2 and Dim==3");
    PHYSIKA_STATIC_ASSERT((is_same<Scalar,float>::value||is_same<Scalar,double>::value||is_same<Scalar,int>::value),
                           "Range<Scalar,Dim> are only defined for Scalar type of float, double and int");
};

}  //end of namespace Physika

#endif  //PHYSIKA_CORE_RANGE_RANGE_H_


