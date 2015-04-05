/*
 * @file array_Nd_iterator-inl.h 
 * @brief  iterator of multi-dimensional array class.
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

#ifndef PHYSIKA_CORE_ARRAYS_ARRAY_ND_ITERATOR_INL_H_
#define PHYSIKA_CORE_ARRAYS_ARRAY_ND_ITERATOR_INL_H_

#include <cstdlib>
#include <iostream>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Vectors/vector_4d.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Utilities/physika_assert.h"

namespace Physika{

template <typename ElementType, int Dim>
ArrayNDIterator<ElementType,Dim>::ArrayNDIterator()
    :array_(NULL),element_idx_(0)
{
}

template <typename ElementType, int Dim>
ArrayNDIterator<ElementType,Dim>::~ArrayNDIterator()
{
}

template <typename ElementType, int Dim>
ArrayNDIterator<ElementType,Dim>::ArrayNDIterator(const ArrayNDIterator<ElementType,Dim> &iterator)
    :array_(iterator.array_),element_idx_(iterator.element_idx_)
{
}

template <typename ElementType, int Dim>
ArrayNDIterator<ElementType,Dim>& ArrayNDIterator<ElementType,Dim>::operator= (const ArrayNDIterator<ElementType,Dim> &iterator)
{
    array_ = iterator.array_;
    element_idx_ = iterator.element_idx_;
    return *this;
}

template <typename ElementType, int Dim>
bool ArrayNDIterator<ElementType,Dim>::operator== (const ArrayNDIterator<ElementType,Dim> &iterator) const
{
    if((array_==NULL)||(iterator.array_==NULL))
        throw PhysikaException("Error: undefined operator == for unintialized iterator!");
    return (element_idx_ == iterator.element_idx_)&&(array_ == iterator.array_);
}

template <typename ElementType, int Dim>
bool ArrayNDIterator<ElementType,Dim>::operator!= (const ArrayNDIterator<ElementType,Dim> &iterator) const
{
    if((array_==NULL)||(iterator.array_==NULL))
        throw PhysikaException("Error: undefined operator != for unintialized iterator!");
    return (element_idx_ != iterator.element_idx_)||(array_ != iterator.array_);
}

template <typename ElementType, int Dim>
ArrayNDIterator<ElementType,Dim>& ArrayNDIterator<ElementType,Dim>::operator++ ()
{
    if(array_==NULL)
        throw PhysikaException("Error: undefined operator ++ for uninitialized iterator!");
    ++element_idx_;
    return *this;
}

template <typename ElementType, int Dim>
ArrayNDIterator<ElementType,Dim>& ArrayNDIterator<ElementType,Dim>::operator-- ()
{
    if(array_==NULL)
        throw PhysikaException("Error: undefined operator -- for uninitialized iterator!");
    --element_idx_;
    return *this;
}

template <typename ElementType, int Dim>
ArrayNDIterator<ElementType,Dim> ArrayNDIterator<ElementType,Dim>::operator++ (int)
{
    if(array_==NULL)
        throw PhysikaException("Error: undefined operator ++ for uninitialized iterator!");
    ArrayNDIterator<ElementType,Dim> iterator(*this);
    ++element_idx_;
    return iterator;
}

template <typename ElementType, int Dim>
ArrayNDIterator<ElementType,Dim> ArrayNDIterator<ElementType,Dim>::operator-- (int)
{
    if(array_==NULL)
        throw PhysikaException("Error: undefined operator -- for uninitialized iterator!");
    ArrayNDIterator<ElementType,Dim> iterator(*this);
    --element_idx_;
    return iterator;
}

template <typename ElementType, int Dim>
ArrayNDIterator<ElementType,Dim> ArrayNDIterator<ElementType,Dim>::operator+ (int stride) const
{
    if(array_==NULL)
        throw PhysikaException("Error: undefined operator + for uninitialized iterator!");
    ArrayNDIterator<ElementType,Dim> iterator(*this);
    iterator.element_idx_ += stride;
    return iterator;
}

template <typename ElementType, int Dim>
ArrayNDIterator<ElementType,Dim> ArrayNDIterator<ElementType,Dim>::operator- (int stride) const
{
    if(array_==NULL)
        throw PhysikaException("Error: undefined operator - for uninitialized iterator!");
    ArrayNDIterator<ElementType,Dim> iterator(*this);
    iterator.element_idx_ -= stride;
    return iterator;
}

template <typename ElementType, int Dim>
const ElementType& ArrayNDIterator<ElementType,Dim>::operator *() const
{
    if(array_==NULL)
        throw PhysikaException("Error: undefined operator * for uninitialized iterator!");
    if(element_idx_ >= array_->totalElementCount())
        throw PhysikaException("Error: iterator out of range!");
    return array_->data_[element_idx_];
}

template <typename ElementType, int Dim>
ElementType& ArrayNDIterator<ElementType,Dim>::operator *()
{
    if(array_==NULL)
        throw PhysikaException("Error: undefined operator * for uninitialized iterator!");
    if(element_idx_ >= array_->totalElementCount())
        throw PhysikaException("Error: iterator out of range!");
    return array_->data_[element_idx_];
}

template <typename ElementType, int Dim>
void ArrayNDIterator<ElementType,Dim>::elementIndex(std::vector<unsigned int> &element_idx) const
{
    if(array_==NULL)
        throw PhysikaException("Error: undefined behavior to get element index for uninitialized iterator!");
    element_idx.resize(Dim);
    unsigned int flat_index = this->element_idx_;
    for(unsigned int i = 0; i < Dim; ++i)
    {
        element_idx[i] = 1;
        for(unsigned int j = i+1; j < Dim; ++j)
            element_idx[i] *= array_->element_count_[j];
        unsigned int temp = flat_index / element_idx[i];
        flat_index = flat_index % element_idx[i];
        element_idx[i] = temp;
    }
}

template <typename ElementType, int Dim>
Vector<unsigned int,Dim> ArrayNDIterator<ElementType,Dim>::elementIndex() const
{
    PHYSIKA_STATIC_ASSERT((Dim==2||Dim==3||Dim==4),"Error: method specific to Dim == 2,3,4!");
    std::vector<unsigned int> element_idx;
    this->elementIndex(element_idx);
    Vector<unsigned int,Dim> result;
    for(unsigned int i = 0; i < Dim; ++i)
        result[i] = element_idx[i];
    return result;
}

}  //end of namespace Physika

#endif  //PHYSIKA_CORE_ARRAYS_ARRAY_ND_ITERATOR_INL_H_
