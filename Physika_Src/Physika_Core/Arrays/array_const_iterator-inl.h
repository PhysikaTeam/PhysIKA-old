/*
 * @file array_const_iterator-inl.h 
 * @brief  const iterator of 1D array class that do not modify its value.
 * @author Fei Zhu
 *
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_ARRAYS_ARRAY_CONST_ITERATOR_INL_H_
#define PHYSIKA_CORE_ARRAYS_ARRAY_CONST_ITERATOR_INL_H_

#include <cstdlib>
#include <iostream>
#include "Physika_Core/Utilities/physika_exception.h"

namespace Physika{

template <typename ElementType>
ArrayConstIterator<ElementType>::ArrayConstIterator()
    :array_(NULL),element_idx_(0)
{
}

template <typename ElementType>
ArrayConstIterator<ElementType>::~ArrayConstIterator()
{
}

template <typename ElementType>
ArrayConstIterator<ElementType>::ArrayConstIterator(const ArrayConstIterator<ElementType> &iterator)
    :array_(iterator.array_),element_idx_(iterator.element_idx_)
{
}

template <typename ElementType>
ArrayConstIterator<ElementType>& ArrayConstIterator<ElementType>::operator= (const ArrayConstIterator<ElementType> &iterator)
{
    array_ = iterator.array_;
    element_idx_ = iterator.element_idx_;
    return *this;
}

template <typename ElementType>
bool ArrayConstIterator<ElementType>::operator== (const ArrayConstIterator<ElementType> &iterator) const
{
    if((array_==NULL)||(iterator.array_==NULL))
        throw PhysikaException("Error: undefined operator == for uninitialized iterator!");
    return (element_idx_ == iterator.element_idx_)&&(array_ == iterator.array_);
}

template <typename ElementType>
bool ArrayConstIterator<ElementType>::operator!= (const ArrayConstIterator<ElementType> &iterator) const
{
    if((array_==NULL)||(iterator.array_==NULL))
        throw PhysikaException("Error: undefined operator != for uninitialized iterator!");
    return (element_idx_ != iterator.element_idx_)||(array_ != iterator.array_);
}

template <typename ElementType>
ArrayConstIterator<ElementType>& ArrayConstIterator<ElementType>::operator++ ()
{
    if(array_==NULL)
        throw PhysikaException("Error: undefined operator ++ for uninitialized iterator!");
    ++element_idx_;
    return *this;
}

template <typename ElementType>
ArrayConstIterator<ElementType>& ArrayConstIterator<ElementType>::operator-- ()
{
    if(array_==NULL)
        throw PhysikaException("Error: undefined operator -- for uninitialized iterator!");
    --element_idx_;
    return *this;
}

template <typename ElementType>
ArrayConstIterator<ElementType> ArrayConstIterator<ElementType>::operator++ (int)
{
    if(array_==NULL)
        throw PhysikaException("Error: undefined operator ++ for uninitialized iterator!");
    ArrayConstIterator<ElementType> iterator(*this);
    ++element_idx_;
    return iterator;
}

template <typename ElementType>
ArrayConstIterator<ElementType> ArrayConstIterator<ElementType>::operator-- (int)
{
    if(array_==NULL)
        throw PhysikaException("Error: undefined operator -- for uninitialized iterator!");
    ArrayConstIterator<ElementType> iterator(*this);
    --element_idx_;
    return iterator;
}

template <typename ElementType>
ArrayConstIterator<ElementType> ArrayConstIterator<ElementType>::operator+ (int stride) const
{
    if(array_==NULL)
        throw PhysikaException("Error: undefined operator + for uninitialized iterator!");
    ArrayConstIterator<ElementType> iterator(*this);
    iterator.element_idx_ += stride;
    return iterator;
}

template <typename ElementType>
ArrayConstIterator<ElementType> ArrayConstIterator<ElementType>::operator- (int stride) const
{
    if(array_==NULL)
        throw PhysikaException("Error: undefined operator - for uninitialized iterator!");
    ArrayConstIterator<ElementType> iterator(*this);
    iterator.element_idx_ -= stride;
    return iterator;
}

template <typename ElementType>
const ElementType& ArrayConstIterator<ElementType>::operator *() const
{
    if(array_==NULL)
        throw PhysikaException("Error: undefined operator * for uninitialized iterator!");
    //leave valid check of index to array class
    return (*array_)[element_idx_];
}

template <typename ElementType>
const ElementType& ArrayConstIterator<ElementType>::operator *()
{
    if(array_==NULL)
        throw PhysikaException("Error: undefined operator * for uninitialized iterator!");
    //leave valid check of index to array class
    return (*array_)[element_idx_];
}

} //end of namespace Physika

#endif //PHYSIKA_CORE_ARRAYS_ARRAY_CONST_ITERATOR_INL_H_
