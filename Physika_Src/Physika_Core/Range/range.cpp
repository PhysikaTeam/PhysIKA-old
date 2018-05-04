/*
 * @file range.cpp
 * @brief higher dimensional counterpart of interval class.
 *        2D example: (1,1) to (3,3) 
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

#include <cstdlib>
#include <iostream>
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Range/range.h"

namespace Physika{

template <typename Scalar,int Dim>
Range<Scalar,Dim>::Range()
    :min_corner_(Vector<Scalar,Dim>(0)),max_corner_(Vector<Scalar,Dim>(0))
{
}

template <typename Scalar,int Dim>
Range<Scalar,Dim>::Range(const Vector<Scalar,Dim> &point)
    :min_corner_(point),max_corner_(point)
{
}

template <typename Scalar,int Dim>
Range<Scalar,Dim>::Range(const Vector<Scalar,Dim> &min_val, const Vector<Scalar,Dim> &max_val)
{
    for(int i = 0; i < Dim; ++i)
        if(min_val[i]>max_val[i])
            throw PhysikaException("Minimum corner of a range must has entries equal or smaller than the maximum corner.");
    min_corner_ = min_val;
    max_corner_ = max_val;
}

template <typename Scalar,int Dim>
Range<Scalar,Dim>::Range(const Range<Scalar,Dim> &range)
    :min_corner_(range.min_corner_),max_corner_(range.max_corner_)
{
}

template <typename Scalar,int Dim>
Range<Scalar,Dim>& Range<Scalar,Dim>::operator= (const Range<Scalar,Dim> &range)
{
    min_corner_ = range.min_corner_;
    max_corner_ = range.max_corner_;
    return *this;
}

template <typename Scalar,int Dim>
bool Range<Scalar,Dim>::operator== (const Range<Scalar,Dim> &range) const
{
    return (min_corner_==range.min_corner_)&&(max_corner_==range.max_corner_);
}

template <typename Scalar,int Dim>
bool Range<Scalar,Dim>::operator!= (const Range<Scalar,Dim> &range) const
{
    return !((*this)==range);
}

template <typename Scalar,int Dim>
Range<Scalar,Dim>::~Range()
{
}

template <typename Scalar,int Dim>
Vector<Scalar,Dim> Range<Scalar,Dim>::center() const
{
    return (max_corner_+min_corner_)/2;
}

template <typename Scalar,int Dim>
Vector<Scalar,Dim> Range<Scalar,Dim>::edgeLengths() const
{
    return max_corner_-min_corner_;
}

template <typename Scalar,int Dim>
Scalar Range<Scalar,Dim>::size() const
{
    Vector<Scalar,Dim> edge_lengths = edgeLengths();
    Scalar result = 1.0;
    for(int i = 0; i < Dim; ++i)
        result *= edge_lengths[i];
    return result;
}

template <typename Scalar,int Dim>
Vector<Scalar,Dim> Range<Scalar,Dim>::minCorner() const
{
    return min_corner_;
}

template <typename Scalar,int Dim>
Vector<Scalar,Dim> Range<Scalar,Dim>::maxCorner() const
{
    return max_corner_;
}

template <typename Scalar,int Dim>
void Range<Scalar,Dim>::setMinCorner(const Vector<Scalar,Dim> &corner)
{    
    min_corner_ = corner;
}

template <typename Scalar,int Dim>
void Range<Scalar,Dim>::setMaxCorner(const Vector<Scalar,Dim> &corner)
{    
    max_corner_ = corner;
}

template <typename Scalar,int Dim>
bool Range<Scalar,Dim>::inside(const Vector<Scalar,Dim> &val) const
{
    for(int i = 0; i < Dim; ++i)
        if(val[i]<min_corner_[i]||val[i]>max_corner_[i])
            return false;
    return true;
}

template <typename Scalar,int Dim>
bool Range<Scalar,Dim>::outside(const Vector<Scalar,Dim> &val) const
{
    return !inside(val);
}

template <typename Scalar,int Dim>
Range<Scalar,Dim> Range<Scalar,Dim>::unitRange()
{
    return Range(Vector<Scalar,Dim>(0),Vector<Scalar,Dim>(1.0));
}

//explicit instantiation
template class Range<unsigned char, 2>;
template class Range<unsigned short, 2>;
template class Range<unsigned int, 2>;
template class Range<unsigned long, 2>;
template class Range<unsigned long long, 2>;
template class Range<signed char, 2>;
template class Range<short, 2>;
template class Range<int, 2>;
template class Range<long, 2>;
template class Range<long long, 2>;
template class Range<float, 2>;
template class Range<double, 2>;
template class Range<long double, 2>;
template class Range<unsigned char, 3>;
template class Range<unsigned short, 3>;
template class Range<unsigned int, 3>;
template class Range<unsigned long, 3>;
template class Range<unsigned long long, 3>;
template class Range<signed char, 3>;
template class Range<short, 3>;
template class Range<int, 3>;
template class Range<long, 3>;
template class Range<long long, 3>;
template class Range<float, 3>;
template class Range<double, 3>;
template class Range<long double, 3>;

}  //end of namespace Physika
