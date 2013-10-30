/*
 * @file array.h 
 * @brief array class. Design for general using. 
 * @author Sheng Yang
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

#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

template <typename ElementType >
class Array
{
public:    
    /* Constructors */
    Array();
    Array(ElementType* data, unsigned int element_count);
    Array(const Array<ElementType>& );
    ~Array();
    
    /* Assignment operators */
    Array<ElementType>& operator = (const Array<ElementType>& arr);

    /* Get and Set functions */
    inline unsigned int elementCount() const { return element_count_; }
    inline ElementType* data() const { return data_; }

    /* Special functions */
    void reset(unsigned int count);
    void setSpace(unsigned int count);
    void zero();

    /* Operator overloading */
    inline ElementType & operator[] (unsigned int id){ assert(id >= 0 && id < element_count_); return data_[id]; }


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
Array<ElementType>::Array(ElementType* data, unsigned int element_count)
{
    setSpace(element_count);
    memcpy(data_,data,sizeof(ElementType)*element_count_);
}

template <typename ElementType>
Array<ElementType>::Array(const Array<ElementType>& arr)
{
    setSpace(arr.elementCount());
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
void Array<ElementType>::reset(const unsigned int count)
{
    release();
    setSpace(count);
}

template <typename ElementType>
void Array<ElementType>::setSpace(const unsigned int count)
{
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
    reset(arr.elementCount());
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
