/*
 * @file array.h 
 * @brief array class. Design for general using. 
 * @author Sheng Yang, Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Core/Array/array.h"

namespace Physika{

template <typename ElementType>
Array<ElementType>::Array():element_count_(0)
{
    data_ = NULL;
}

template <typename ElementType>
Array<ElementType>::Array(unsigned int element_count)
{
    data_ = NULL;
    resize(element_count);
}

template <typename ElementType>
Array<ElementType>::Array(unsigned int element_count, const ElementType &value)
{
    data_ = NULL;
    resize(element_count);
    for (int i = 0; i < element_count_; ++i)
	data_[i] = value;
}

template <typename ElementType>
Array<ElementType>::Array(unsigned int element_count, const ElementType *data)
{
    data_ = NULL;
    resize(element_count);
    memcpy(data_, data, sizeof(ElementType) * element_count_);
}

template <typename ElementType>
Array<ElementType>::Array(const Array<ElementType> &arr)
{
    data_ = NULL;
    resize(arr.elementCount());
    memcpy(data_, arr.data(), sizeof(ElementType) * element_count_);
}

template <typename ElementType>
Array<ElementType>::~Array()
{
    release();
}

template <typename ElementType>
void Array<ElementType>::allocate()
{
    data_ = new ElementType[element_count_];
}

template <typename ElementType>
void Array<ElementType>::release()
{
    if(data_ != NULL)
        delete [] data_;
    data_ = NULL;
}

template <typename ElementType>
void Array<ElementType>::resize(unsigned int count)
{
    release();
    element_count_ = count;
    allocate();
}

template <typename ElementType>
void Array<ElementType>::zero()
{
    memset((void*)data_, 0, element_count_ * sizeof(ElementType));
}

template <typename ElementType>
Array<ElementType> & Array<ElementType>::operator = (const Array<ElementType> &arr)
{
    resize(arr.elementCount());
    memcpy(data_,arr.data(), sizeof(ElementType) * element_count_);
    return *this;
}

template <typename ElementType>
void Array<ElementType>::Reordering(unsigned int *ids, unsigned int size)
{
    if (size != element_count_)
    {
        std::cout << "array size do not match!" << std::endl;
        exit(0);
    }

    ElementType * tmp = new ElementType[element_count_];
    for (size_t i = 0; i < element_count_; i++)
    {
        tmp[i] = data_[ids[i]];
    }

    memcpy(data_, tmp, element_count_ * sizeof(ElementType));

}

template class Array<float>;
template class Array<double>;
template class Array<int>;

}