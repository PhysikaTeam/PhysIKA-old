/*
 * @file array-inl.h 
 * @brief implementation of methods in array.h
 * @author Sheng Yang, Fei Zhu
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_ARRAYS_ARRAY_INL_H_
#define PHYSIKA_CORE_ARRAYS_ARRAY_INL_H_

#include <cstdlib>

namespace Physika{


template <typename ElementType>
std::ostream & operator<< (std::ostream &s, const Array<ElementType> &arr)
{
    s<<"[";
    for (unsigned int i = 0; i < arr.elementCount(); i++)
    {
        s<<arr[i];
        if(i != arr.elementCount()-1)
            s<<", ";
    }
    s<<"]";
    return s; 
}

template <typename ElementType>
Array<ElementType>::Array():element_count_(0),data_(NULL)
{
}

template <typename ElementType>
Array<ElementType>::Array(unsigned int element_count):data_(NULL)
{
    resize(element_count);
}

template <typename ElementType>
Array<ElementType>::Array(unsigned int element_count, const ElementType &value)
    :data_(NULL)
{
    resize(element_count);
    memset(data_, value, sizeof(ElementType)*element_count_);
}

template <typename ElementType>
Array<ElementType>::Array(unsigned int element_count, const ElementType *data)
    :data_(NULL)
{
    resize(element_count);
    memcpy(data_, data, sizeof(ElementType) * element_count_);
}

template <typename ElementType>
Array<ElementType>::Array(const Array<ElementType> &arr)
    :data_(NULL)
{
    resize(arr.elementCount());
    memcpy(data_, arr.data(), sizeof(ElementType) * element_count_);
}

template <typename ElementType>
Array<ElementType>::~Array()
{
    release();
}

template <typename ElementType>
Array<ElementType>& Array<ElementType>::operator = (const Array<ElementType> &arr)
{
    resize(arr.elementCount());
    memcpy(data_,arr.data(), sizeof(ElementType) * element_count_);
    return *this;
}

template <typename ElementType>
void Array<ElementType>::resize(unsigned int count)
{
    release();
    element_count_ = count;
    allocate();
}

template <typename ElementType>
ElementType& Array<ElementType>::operator[] (unsigned int id)
{
    if(id<0||id>=element_count_)
    {
        std::cerr<<"Array index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return data_[id];
}

template <typename ElementType>
ElementType& Array<ElementType>::operator[] (unsigned int id) const
{
    if(id<0||id>=element_count_)
    {
        std::cerr<<"Array index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return data_[id];
}

template <typename ElementType>
void Array<ElementType>::permutate(unsigned int *ids, unsigned int size)
{
    if(size != element_count_)
    {
        std::cerr << "array size do not match!" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if(element_count_>0)
    {
        ElementType * tmp = new ElementType[element_count_];
        PHYSIKA_ASSERT(tmp);
        PHYSIKA_ASSERT(data_);
        for (size_t i = 0; i < element_count_; i++)
            tmp[i] = data_[ids[i]];
        memcpy(data_, tmp, element_count_ * sizeof(ElementType));
        delete[] tmp;
    }
}

template <typename ElementType>
void Array<ElementType>::allocate()
{
    if(data_ != NULL)
        delete[] data_;
    data_ = new ElementType[element_count_];
    PHYSIKA_ASSERT(data_);
}

template <typename ElementType>
void Array<ElementType>::release()
{
    if(data_ != NULL)
        delete[] data_;
    data_ = NULL;
}

}  //end of namespace Physika

#endif  //PHYSIKA_CORE_ARRAYS_ARRAY_INL_H_
