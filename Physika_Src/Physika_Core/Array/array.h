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

#ifndef PHSYIKA_CORE_ARRAY_ARRAY_H_
#define PHSYIKA_CORE_ARRAY_ARRAY_H_

#include <cstring>
#include "Physika_Core/Utilities/physika_assert.h"

namespace Physika{

template <typename ElementType >
class Array
{
public:    
    /* Constructors */
    Array();//empty array
    explicit Array(unsigned int element_count);//array with given size, uninitialized value
    Array(unsigned int element_count,const ElementType &value);//array with given size, initialized with same value
    Array(unsigned int element_count,const ElementType* data);//array with given size, initialized with given data
    Array(const Array<ElementType>& );
    ~Array();
    
    /* Assignment operators */
    Array<ElementType>& operator = (const Array<ElementType>& arr);

    /* Get and Set functions */
    inline unsigned int elementCount() const { return element_count_; }
    inline ElementType* data() const { return data_; }

    /* Special functions */
    void resize(unsigned int count);
    void zero();

    /* Operator overloading */
    inline ElementType & operator[] (unsigned int id){ PHYSIKA_ASSERT(id >= 0 && id < element_count_); return data_[id]; }

protected:
    void allocate();
    void release();

    unsigned int element_count_;
    ElementType * data_;

};


template <typename ElementType>
Array<ElementType>::Array():element_count_(0)
{
    data_ = NULL;
}

template <typename ElementType>
Array<ElementType>::Array(unsigned int element_count)
{
    resize(element_count);
}

template <typename ElementType>
Array<ElementType>::Array(unsigned int element_count,const ElementType &value)
{
    resize(element_count);
    for(int i = 0; i < element_count_;++i)
	data_[i] = value;
}

template <typename ElementType>
Array<ElementType>::Array(unsigned int element_count,const ElementType* data)
{
    resize(element_count);
    memcpy(data_,data,sizeof(ElementType)*element_count_);
}

template <typename ElementType>
Array<ElementType>::Array(const Array<ElementType>& arr)
{
    resize(arr.elementCount());
    memcpy(data_,arr.data(),sizeof(ElementType)*element_count_);
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
    memset((void*)data_, 0, element_count_*sizeof(ElementType));
}

template <typename ElementType>
Array<ElementType>& Array<ElementType>::operator = (const Array<ElementType>& arr)
{
    resize(arr.elementCount());
    memcpy(data_,arr.data(),sizeof(ElementType)*element_count_);
    return *this;
}

template <typename ElementType>
std::ostream& operator<< (std::ostream &s, const Array<ElementType> &arr)
{
    for(size_t i = 0; i < arr.elementCount(); i++)
    {
        if(i == 0)
            s<<arr[i];
        s<<", "<<arr[i];
    }
    s<<std::endl;
    return s; 
}

}//end of namespace Physika

#endif //PHSYIKA_CORE_ARRAY_ARRAY_H_
