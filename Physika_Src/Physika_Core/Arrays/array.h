/*
 * @file array.h 
 * @brief 1D array class. Designed for general use.
 * @author Sheng Yang, Fei Zhu
 * @Suggestion: Choose between Array and std::vector at your will.
 *              If frequent sort and find operation is needed, use std::vector and its algorithm.
 *              Otherwise, you could give Array a try.
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHSYIKA_CORE_ARRAYS_ARRAY_H_
#define PHSYIKA_CORE_ARRAYS_ARRAY_H_

#include <cstring>
#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"

namespace Physika{

/*
 * ReorderObject: base class of Array such that object of Array class can be pointed by
 * template-free pointer ReorderObject*. For use in ArrayManager, which permutates elements
 * of arrays concurrently, see array_manager.h.
 */

class ReorderObject
{
public:
    ReorderObject() {}
    ~ReorderObject() {}
    virtual void reorder(unsigned int *ids, unsigned int size) = 0;
};

template <typename ElementType >
class Array: public ReorderObject
{
public:    
    /* Constructors */
    Array();//empty array
    explicit Array(unsigned int element_count);//array with given size, uninitialized value
    Array(unsigned int element_count, const ElementType &value);//array with given size, initialized with same value
    Array(unsigned int element_count, const ElementType *data);//array with given size, initialized with given data
    Array(const Array<ElementType> &);
    ~Array();
    
    /* Assignment operators */
    Array<ElementType> & operator = (const Array<ElementType> &arr);

    /* Get and Set functions */
    inline unsigned int elementCount() const { return element_count_; }
    inline unsigned int size() const { return element_count_; }
    inline ElementType* data() const { return data_; }

    /* Special functions */
    void resize(unsigned int count);
    void zero();

    /* Operator overloading */
    ElementType & operator[] (int id){ PHYSIKA_ASSERT(id >= 0 && id < element_count_); return data_[id]; }

    virtual void reorder(unsigned int *ids, unsigned int size);
protected:
    void allocate();
    void release();

    unsigned int element_count_;
    ElementType *data_;

};

//implementation
//Array is designed for general use, it's impossible to declare all ElementType with explicit instantiation
//thus implementation is put in header file
template <typename ElementType>
std::ostream & operator<< (std::ostream &s, const Array<ElementType> &arr)
{
    s<<"[";
    for (size_t i = 0; i < arr.elementCount(); i++)
    {
	s<<arr[i];
	if(i != arr.elementCount()-1)
	    s<<", "<<1;
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
    for (int i = 0; i < element_count_; ++i)
	data_[i] = value;
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
void Array<ElementType>::reorder(unsigned int *ids, unsigned int size)
{
    if (size != element_count_)
    {
        std::cerr << "array size do not match!" << std::endl;
	std::exit(EXIT_FAILURE);
    }

    ElementType * tmp = new ElementType[element_count_];
    for (size_t i = 0; i < element_count_; i++)
        tmp[i] = data_[ids[i]];
    memcpy(data_, tmp, element_count_ * sizeof(ElementType));
    delete[] tmp;
}

}//end of namespace Physika

#endif //PHSYIKA_CORE_ARRAYS_ARRAY_H_
