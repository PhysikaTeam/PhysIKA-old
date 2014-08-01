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
 * ArrayBase: base class of Array, provides virtual methods implemented by Array class.
 * In this way object of Array class can be pointed by template-free pointer ArrayBase*.
 * For use in ArrayManager, which performs operations  on elements of several arrays 
 * concurrently, see array_manager.h.
 */

class ArrayBase
{
public:
    ArrayBase() {}
    ~ArrayBase() {}
    //perform permutation of the array's elements
    virtual void permutate(unsigned int *ids, unsigned int size) = 0;
};

template <typename ElementType >
class Array: public ArrayBase
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
    Array<ElementType>& operator = (const Array<ElementType> &arr);

    /* Get and Set functions */
    inline unsigned int elementCount() const { return element_count_; }
    inline unsigned int size() const { return element_count_; }
    inline ElementType* data() const { return data_; }

    /* Special functions */
    void resize(unsigned int count);    //resize array, data will be lost

    /* Operator overloading */
    ElementType & operator[] (unsigned int id);
    ElementType & operator[] (unsigned int id) const;

    virtual void permutate(unsigned int *ids, unsigned int size);
    
    void release();

protected:
    void allocate();
    

    unsigned int element_count_;
    ElementType *data_;
};

}//end of namespace Physika

//implementation
//Array is designed for general use, it's impossible to declare all ElementType with explicit instantiation
//thus implementation is put in header file, we put it in another header file to make array interface clear
#include "Physika_Core/Arrays/array-inl.h"

#endif //PHSYIKA_CORE_ARRAYS_ARRAY_H_
