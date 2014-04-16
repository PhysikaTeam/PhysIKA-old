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
#include <list>
#include <map>
#include <algorithm>
#include <iostream>
#include <string>
#include "Physika_Core/Utilities/physika_assert.h"

namespace Physika{

    class ReorderObject
    {
    public:
        ReorderObject() {};
        ~ReorderObject() {};
        virtual void Reordering(unsigned int *ids, unsigned int size) = 0;
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
    inline ElementType * data() const { return data_; }

    /* Special functions */
    void resize(unsigned int count);
    void zero();

    /* Operator overloading */
    ElementType & operator[] (int id){ PHYSIKA_ASSERT(id >= 0 && id < element_count_); return data_[id]; }

    virtual void Reordering(unsigned int *ids, unsigned int size);
protected:
    void allocate();
    void release();

    unsigned int element_count_;
    ElementType *data_;

};




//std::ostream& operator<< (std::ostream &s, const Vector<Scalar,2> &vec)
template <typename ElementType>
std::ostream & operator<< (std::ostream &s, const Array<ElementType> &arr)
{
    for (size_t i = 0; i < arr.elementCount(); i++)
    {
        if (i == 0)
            s<<arr[i];
        s<<", "<<1;
    }
    s<<std::endl;
    return s; 
}





}//end of namespace Physika

#endif //PHSYIKA_CORE_ARRAY_ARRAY_H_
