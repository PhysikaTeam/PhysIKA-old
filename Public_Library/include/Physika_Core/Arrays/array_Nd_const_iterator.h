/*
 * @file array_Nd_const_iterator.h 
 * @brief  const iterator of multi-dimensional array class.
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

#ifndef PHYSIKA_CORE_ARRAYS_ARRAY_ND_CONST_ITERATOR_H_
#define PHYSIKA_CORE_ARRAYS_ARRAY_ND_CONST_ITERATOR_H_

#include <vector>

namespace Physika{

template <typename ElementType, int Dim> class ArrayND;

template <typename ElementType, int Dim>
class ArrayNDConstIterator
{
public:
    ArrayNDConstIterator();
    ~ArrayNDConstIterator();
    ArrayNDConstIterator(const ArrayNDConstIterator<ElementType,Dim> &iterator);
    ArrayNDConstIterator<ElementType,Dim>& operator= (const ArrayNDConstIterator<ElementType,Dim> &iterator);
    bool operator== (const ArrayNDConstIterator<ElementType,Dim> &iterator) const;
    bool operator!= (const ArrayNDConstIterator<ElementType,Dim> &iterator) const;
    ArrayNDConstIterator<ElementType,Dim>& operator++ ();
    ArrayNDConstIterator<ElementType,Dim>& operator-- ();
    ArrayNDConstIterator<ElementType,Dim> operator++ (int);
    ArrayNDConstIterator<ElementType,Dim> operator-- (int);
    ArrayNDConstIterator<ElementType,Dim> operator+ (int stride) const;
    ArrayNDConstIterator<ElementType,Dim> operator- (int stride) const;
    const ElementType& operator *() const;
    const ElementType& operator *();
    //return the index of the elment pointed to
    void elementIndex(std::vector<unsigned int> &element_idx) const; 
    Vector<unsigned int,Dim> elementIndex() const; //specific to Dim = 2,3,4
protected:
    const ArrayND<ElementType,Dim> *array_;
    unsigned int element_idx_;
    friend class ArrayND<ElementType,Dim>;
};

}  //end of namespace Physika

//implementation
#include "Physika_Core/Arrays/array_Nd_const_iterator-inl.h"

#endif //PHYSIKA_CORE_ARRAYS_ARRAY_ND_CONST_ITERATOR_H_
