/*
 * @file array_Nd-inl.h 
 * @brief  Implementation of methods in array_Nd.h.
 * @author Fei Zhu
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_ARRAYS_ARRAY_ND_INL_H_
#define PHYSIKA_CORE_ARRAYS_ARRAY_ND_INL_H_

#include "Physika_Core/Utilities/physika_exception.h"

namespace Physika{

template <typename ElementType,int Dim>
ArrayND<ElementType,Dim>::ArrayND():data_(NULL)
{
    PHYSIKA_STATIC_ASSERT(Dim>1,"ArrayND are defined for dimension higher than 1");
	for(unsigned int i = 0; i < Dim; ++i)
		element_count_[i] = 0;
}

template <typename ElementType,int Dim>
ArrayND<ElementType,Dim>::ArrayND(const std::vector<unsigned int> &element_counts):data_(NULL)
{
    PHYSIKA_STATIC_ASSERT(Dim>1,"ArrayND are defined for dimension higher than 1");
    if(element_counts.size()!=Dim)
        throw PhysikaException("Dimension of element counts mismatches the dimension of array!");
    resize(element_counts);
}

template <typename ElementType,int Dim>
ArrayND<ElementType,Dim>::ArrayND(const std::vector<unsigned int> &element_counts, const ElementType &value)
    :data_(NULL)
{
    PHYSIKA_STATIC_ASSERT(Dim>1,"ArrayND are defined for dimension higher than 1");
    if(element_counts.size()!=Dim)
        throw PhysikaException("Dimension of element counts mismatches the dimension of array!");
    resize(element_counts);
    unsigned int total_count = totalElementCount();
    for(unsigned int i = 0; i < total_count; ++i)
        data_[i] = value;
}

template <typename ElementType,int Dim>
    ArrayND<ElementType,Dim>::ArrayND(const Vector<unsigned int,Dim> &element_counts):data_(NULL)
{
    PHYSIKA_STATIC_ASSERT((Dim==2||Dim==3||Dim==4),"Error: method specific to Dim == 2,3,4!");
    std::vector<unsigned int> counts;
    for(unsigned int i = 0; i < Dim; ++i)
        counts.push_back(element_counts[i]);
    resize(counts);
}

template <typename ElementType,int Dim>
ArrayND<ElementType,Dim>::ArrayND(const Vector<unsigned int,Dim> &element_counts, const ElementType &value)
    :data_(NULL)
{
    PHYSIKA_STATIC_ASSERT((Dim==2||Dim==3||Dim==4),"Error: method specific to Dim == 2,3,4!");
    std::vector<unsigned int> counts;
    for(unsigned int i = 0; i < Dim; ++i)
        counts.push_back(element_counts[i]);
    resize(counts);
    unsigned int total_count = totalElementCount();
    for(unsigned int i = 0; i < total_count; ++i)
        data_[i] = value;
}

template <typename ElementType,int Dim>
ArrayND<ElementType,Dim>::ArrayND(const ArrayND<ElementType,Dim> &array)
    :data_(NULL)
{
    std::vector<unsigned int> element_counts = array.size();
    resize(element_counts);
    unsigned int total_count = totalElementCount();
    memcpy(data_,array.data_,sizeof(ElementType)*total_count);
}

template <typename ElementType,int Dim>
ArrayND<ElementType,Dim>::~ArrayND()
{
    release();
}

template <typename ElementType,int Dim>
ArrayND<ElementType,Dim>& ArrayND<ElementType,Dim>::operator= (const ArrayND<ElementType,Dim> &array)
{
    std::vector<unsigned int> element_counts = array.size();
    resize(element_counts);
    unsigned int total_count = totalElementCount();
    memcpy(data_,array.data_,sizeof(ElementType)*total_count);
    return *this;
}

template <typename ElementType,int Dim>
unsigned int ArrayND<ElementType,Dim>::elementCount(unsigned int dim) const
{
    if(dim>=Dim)
        throw PhysikaException("Dimension out of range!");
    return element_count_[dim];
}

template <typename ElementType,int Dim>
unsigned int ArrayND<ElementType,Dim>::size(unsigned int dim) const
{
    return elementCount(dim);
}

template <typename ElementType,int Dim>
void ArrayND<ElementType,Dim>::elementCount(std::vector<unsigned int> &count) const
{
    count.resize(Dim);
    for(unsigned int i = 0; i < count.size(); ++i)
        count[i] = element_count_[i];
}

template <typename ElementType,int Dim>
void ArrayND<ElementType,Dim>::size(std::vector<unsigned int> &count) const
{
    elementCount(count);
}

template <typename ElementType,int Dim>
void ArrayND<ElementType,Dim>::clear()
{
    release();
	for(unsigned int i = 0; i < Dim; ++i)
		element_count_[i] = 0;
}

template <typename ElementType,int Dim>
void ArrayND<ElementType,Dim>::resize(unsigned int count, unsigned int dim)
{
    if(dim>=Dim)
        throw PhysikaException("Dimension out of range!");
    element_count_[dim] = count;
    allocate();
}

template <typename ElementType,int Dim>
void ArrayND<ElementType,Dim>::resize(const std::vector<unsigned int> &count)
{
    if(count.size()!=Dim)
        throw PhysikaException("Dimension of element counts mismatches the dimension of array!");
    for(unsigned int i = 0; i < count.size(); ++i)
        element_count_[i] = count[i];
    allocate();
}

template <typename ElementType,int Dim>
ElementType& ArrayND<ElementType,Dim>::operator() (const std::vector<unsigned int> &idx)
{
    return elementAtIndex(idx);
}

template <typename ElementType,int Dim>
const ElementType& ArrayND<ElementType,Dim>::operator() (const std::vector<unsigned int> &idx) const
{
    return elementAtIndex(idx);
}

template <typename ElementType,int Dim>
ElementType& ArrayND<ElementType,Dim>::elementAtIndex(const std::vector<unsigned int> &idx)
{
    if(idx.size()!=Dim)
        throw PhysikaException("Dimension of index mismatches the dimension of array!");
    for(unsigned int i = 0; i < idx.size(); ++i)
    {
        if(idx[i]>=element_count_[i])
            throw PhysikaException("Array index out of range!");
    }
    unsigned int index_1d = index1D(idx);
    return data_[index_1d];
}

template <typename ElementType,int Dim>
const ElementType& ArrayND<ElementType,Dim>::elementAtIndex(const std::vector<unsigned int> &idx) const
{
    if(idx.size()!=Dim)
        throw PhysikaException("Dimension of index mismatches the dimension of array!");
    for(unsigned int i = 0; i < idx.size(); ++i)
    {
        if(idx[i]>=element_count_[i])
            throw PhysikaException("Array index out of range!");
    }
    unsigned int index_1d = index1D(idx);
    return data_[index_1d];
}

template <typename ElementType,int Dim>
Vector<unsigned int,Dim> ArrayND<ElementType,Dim>::elementCount() const
{
    PHYSIKA_STATIC_ASSERT((Dim==2||Dim==3||Dim==4),"Error: method specific to Dim == 2,3,4!");
    Vector<unsigned int,Dim> element_count;
    for(unsigned int i = 0; i < Dim; ++i)
        element_count[i] = element_count_[i];
    return element_count;
}

template <typename ElementType,int Dim>
Vector<unsigned int,Dim> ArrayND<ElementType,Dim>::size() const
{
    return elementCount();
}

template <typename ElementType,int Dim>
void ArrayND<ElementType,Dim>::resize(const Vector<unsigned int,Dim> &count)
{
    PHYSIKA_STATIC_ASSERT((Dim==2||Dim==3||Dim==4),"Error: method specific to Dim == 2,3,4!");
    for(unsigned int i = 0; i < Dim; ++i)
        element_count_[i] = count[i];
    allocate();
}

template <typename ElementType,int Dim>
ElementType& ArrayND<ElementType,Dim>::operator() (const Vector<unsigned int,Dim> &idx)
{
    PHYSIKA_STATIC_ASSERT((Dim==2||Dim==3||Dim==4),"Error: method specific to Dim == 2,3,4!");
    return elementAtIndex(idx);
}

template <typename ElementType,int Dim>
const ElementType& ArrayND<ElementType,Dim>::operator() (const Vector<unsigned int,Dim> &idx) const
{
    PHYSIKA_STATIC_ASSERT((Dim==2||Dim==3||Dim==4),"Error: method specific to Dim == 2,3,4!");
    return elementAtIndex(idx);
}

template <typename ElementType,int Dim>
ElementType& ArrayND<ElementType,Dim>::elementAtIndex(const Vector<unsigned int,Dim> &idx)
{
    PHYSIKA_STATIC_ASSERT((Dim==2||Dim==3||Dim==4),"Error: method specific to Dim == 2,3,4!");
    std::vector<unsigned int> idx_vec;
    for(unsigned int i = 0; i < Dim; ++i)
    {
        if(idx[i]>=element_count_[i])
            throw PhysikaException("Array index out of range!");
        idx_vec.push_back(idx[i]);
    }
    unsigned int index_1d = index1D(idx_vec);
    return data_[index_1d];
}

template <typename ElementType,int Dim>
const ElementType& ArrayND<ElementType,Dim>::elementAtIndex(const Vector<unsigned int,Dim> &idx) const
{
    PHYSIKA_STATIC_ASSERT((Dim==2||Dim==3||Dim==4),"Error: method specific to Dim == 2,3,4!");
    std::vector<unsigned int> idx_vec;
    for(unsigned int i = 0; i < Dim; ++i)
    {
        if(idx[i]>=element_count_[i])
            throw PhysikaException("Array index out of range!");
        idx_vec.push_back(idx[i]);
    }
    unsigned int index_1d = index1D(idx_vec);
    return data_[index_1d];
}

template <typename ElementType,int Dim>
void ArrayND<ElementType,Dim>::allocate()
{
    if(data_)
        delete[] data_;
    unsigned int total_count = totalElementCount();
    data_ = new ElementType[total_count];
    PHYSIKA_ASSERT(data_);
}

template <typename ElementType,int Dim>
typename ArrayND<ElementType,Dim>::Iterator ArrayND<ElementType,Dim>::begin()
{
    ArrayND<ElementType,Dim>::Iterator iter;
    iter.array_ = this;
    iter.element_idx_ = 0;
    return iter;
}

template <typename ElementType,int Dim>
typename ArrayND<ElementType,Dim>::Iterator ArrayND<ElementType,Dim>::end()
{
    ArrayND<ElementType,Dim>::Iterator iter;
    iter.array_ = this;
    iter.element_idx_ = this->totalElementCount();
    return iter;
}

template <typename ElementType,int Dim>
typename ArrayND<ElementType,Dim>::ConstIterator ArrayND<ElementType,Dim>::begin() const
{
    ArrayND<ElementType,Dim>::ConstIterator iter;
    iter.array_ = this;
    iter.element_idx_ = 0;
    return iter;
}

template <typename ElementType,int Dim>
typename ArrayND<ElementType,Dim>::ConstIterator ArrayND<ElementType,Dim>::end() const
{
    ArrayND<ElementType,Dim>::ConstIterator iter;
    iter.array_ = this;
    iter.element_idx_ = this->totalElementCount();
    return iter;
}

template <typename ElementType,int Dim>
void ArrayND<ElementType,Dim>::release()
{
    if(data_)
        delete[] data_;
    data_ = NULL;
}

template <typename ElementType,int Dim>
unsigned int ArrayND<ElementType,Dim>::totalElementCount() const
{
    unsigned int total_count = 1;
    for(unsigned int i = 0; i < Dim; ++i)
        total_count *= element_count_[i];
    return total_count;
}

template <typename ElementType,int Dim>
unsigned int ArrayND<ElementType,Dim>::index1D(const std::vector<unsigned int> &idx) const
{
    PHYSIKA_ASSERT(idx.size()==Dim);
    for(unsigned int i = 0; i < Dim; ++i)
        PHYSIKA_ASSERT(idx[i]>=0&&idx[i]<element_count_[i]);
    unsigned int index = 0;
    for(unsigned int i = 0; i < Dim; ++i)
    {
        unsigned int temp = idx[i];
        for(unsigned int j = i+1; j < Dim; ++j)
            temp *= element_count_[j];
        index += temp;
    }
    return index;
}

}  //end of namespace Physika

#endif //PHYSIKA_CORE_ARRAYS_ARRAY_ND_INL_H_
