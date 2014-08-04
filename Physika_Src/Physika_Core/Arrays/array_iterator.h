/*
 * @file array_iterator.h 
 * @brief  iterator of 1D array class.
 * @author Fei Zhu
 *
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_ARRAYS_ARRAY_ITERATOR_H_
#define PHYSIKA_CORE_ARRAYS_ARRAY_ITERATOR_H_

namespace Physika{

template <typename ElementType> class Array;

template <typename ElementType>
class ArrayIterator
{
public:
    ArrayIterator();
    ~ArrayIterator();
    ArrayIterator(const ArrayIterator<ElementType> &iterator);
    ArrayIterator<ElementType>& operator= (const ArrayIterator<ElementType> &iterator);
    bool operator== (const ArrayIterator<ElementType> &iterator) const;
    bool operator!= (const ArrayIterator<ElementType> &iterator) const;
    ArrayIterator<ElementType>& operator++ ();
    ArrayIterator<ElementType>& operator-- ();
    ArrayIterator<ElementType> operator++ (int);
    ArrayIterator<ElementType> operator-- (int);
    ArrayIterator<ElementType> operator+ (int stride) const;
    ArrayIterator<ElementType> operator- (int stride) const;
    const ElementType& operator *() const;
    ElementType& operator *(); 
protected:
    Array<ElementType> *array_;
    unsigned int element_idx_;
    friend class Array<ElementType>;
};

}  //end of namespace Physika

//implementation
#include "Physika_Core/Arrays/array_iterator-inl.h"

#endif //PHYSIKA_CORE_ARRAYS_ARRAY_ITERATOR_H_
