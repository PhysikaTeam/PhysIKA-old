/*
 * @file array_const_iterator.h 
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

#ifndef PHYSIKA_CORE_ARRAYS_ARRAY_CONST_ITERATOR_H_
#define PHYSIKA_CORE_ARRAYS_ARRAY_CONST_ITERATOR_H_

namespace Physika{

template <typename ElementType> class Array;

template <typename ElementType>
class ArrayConstIterator
{
public:
    ArrayConstIterator();
    ~ArrayConstIterator();
    ArrayConstIterator(const ArrayConstIterator<ElementType> &iterator);
    ArrayConstIterator<ElementType>& operator= (const ArrayConstIterator<ElementType> &iterator);
    bool operator== (const ArrayConstIterator<ElementType> &iterator) const;
    bool operator!= (const ArrayConstIterator<ElementType> &iterator) const;
    ArrayConstIterator<ElementType>& operator++ ();
    ArrayConstIterator<ElementType>& operator-- ();
    ArrayConstIterator<ElementType> operator++ (int);
    ArrayConstIterator<ElementType> operator-- (int);
    ArrayConstIterator<ElementType> operator+ (int stride) const;
    ArrayConstIterator<ElementType> operator- (int stride) const;
    const ElementType& operator *() const;
    const ElementType& operator *();   
protected:
    const Array<ElementType> *array_;
    unsigned int element_idx_;
    friend class Array<ElementType>;
};

}  //end of namespace Physika

//implementation
#include "Physika_Core/Arrays/array_const_iterator-inl.h"

#endif //PHYSIKA_CORE_ARRAYS_ARRAY_CONST_ITERATOR_H_
