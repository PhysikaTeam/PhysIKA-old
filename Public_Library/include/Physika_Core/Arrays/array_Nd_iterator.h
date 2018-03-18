/*
 * @file array_Nd_iterator.h 
 * @brief  iterator of multi-dimensional array class.
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

#ifndef PHYSIKA_CORE_ARRAYS_ARRAY_ND_ITERATOR_H_
#define PHYSIKA_CORE_ARRAYS_ARRAY_ND_ITERATOR_H_

#include <vector>

namespace Physika{

template <typename ElementType, int Dim> class ArrayND;

template <typename ElementType, int Dim>
class ArrayNDIterator
{
public:
    ArrayNDIterator();
    ~ArrayNDIterator();
    ArrayNDIterator(const ArrayNDIterator<ElementType,Dim> &iterator);
    ArrayNDIterator<ElementType,Dim>& operator= (const ArrayNDIterator<ElementType,Dim> &iterator);
    bool operator== (const ArrayNDIterator<ElementType,Dim> &iterator) const;
    bool operator!= (const ArrayNDIterator<ElementType,Dim> &iterator) const;
    ArrayNDIterator<ElementType,Dim>& operator++ ();
    ArrayNDIterator<ElementType,Dim>& operator-- ();
    ArrayNDIterator<ElementType,Dim> operator++ (int);
    ArrayNDIterator<ElementType,Dim> operator-- (int);
    ArrayNDIterator<ElementType,Dim> operator+ (int stride) const;
    ArrayNDIterator<ElementType,Dim> operator- (int stride) const;
    const ElementType& operator *() const;
    ElementType& operator *();
    //return the index of the elment pointed to
    void elementIndex(std::vector<unsigned int> &element_idx) const; 
    Vector<unsigned int,Dim> elementIndex() const; //specific to Dim = 2,3,4
protected:
    ArrayND<ElementType,Dim> *array_;
    unsigned int element_idx_;
    friend class ArrayND<ElementType,Dim>;
};

}  //end of namespace Physika

//implementation
#include "Physika_Core/Arrays/array_Nd_iterator-inl.h"

#endif //PHYSIKA_CORE_ARRAYS_ARRAY_ND_ITERATOR_H_
