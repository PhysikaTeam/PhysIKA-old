/*
 * @file array_Nd.h 
 * @brief  Multi-dimensional array class.
 * @author Fei Zhu
 * @Suggestion: Choose between Array and std::vector at your will.
 *              If frequent sort and find operation is needed, use std::vector.
 *              Otherwise, you could give Array a try.
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_ARRAYS_ARRAY_ND_H_
#define PHYSIKA_CORE_ARRAYS_ARRAY_ND_H_

#include <cstring>
#include <vector>
#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"

namespace Physika{

/*
 * ArrayND<ElementType,Dim>: Dim-dimensional array with elements of ElementType
 * If Dim==1, it's 1D array defined array.h
 * We separate 1D array in Array class because we want to keep the simple
 * form of Array<ElementType> for 1D array instead of Array<ElementType,1>
 * Hence:
 *       Use Array<ElementType> for 1D array and ArrayND<ElmentType,Dim> for higer dimension
 *       ArrayND<ElementType,1> will result in compiler error due to the static assert
 */

template <typename ElementType,int Dim>
class ArrayND
{
public:
    ArrayND();  //empty array
    explicit ArrayND(const std::vector<unsigned int> &element_counts);  //array with given size in each dimension, uninitialized value
    ArrayND(const std::vector<unsigned int> &element_counts, const ElementType &value); //array with given size in each dimension, initialized with same value
    ArrayND(const ArrayND<ElementType,Dim> &);
    ~ArrayND();

    ArrayND<ElementType,Dim>& operator= (const ArrayND<ElementType,Dim> &);
    unsigned int elementCount(unsigned int dim) const;  //element count of given dimension
    unsigned int size(unsigned int dim) const;
    std::vector<unsigned int> elementCount() const;  //element count of all dimensions
    std::vector<unsigned int> size() const;
    //resize array, data will be lost
    void resize(unsigned int count, unsigned int dim);  //resize given dimension
    void resize(const std::vector<unsigned int> &count);  //resize all dimensions
    ElementType& operator() (const std::vector<unsigned int> &idx); //get element at given index
    ElementType& elementAtIndex(const std::vector<unsigned int> &idx);
protected:
    void allocate();
    void release();
    unsigned int totalElementCount() const;  //return the total number of elements
    unsigned int index1D(const std::vector<unsigned int> &idx) const; //given high dimension index, return 1d version
protected:
    unsigned int elment_count_[Dim];  //number of element in each dimension
    ElementType *data_;  //data stored in 1D
protected:
    PHYSIKA_STATIC_ASSERT(Dim>1,"ArrayND are defined for dimension higher than 1");
};

}  //end of namespace Physika

//implementation
#include "Physika_Core/Arrays/array_Nd-inl.h"

#endif //PHYSIKA_CORE_ARRAYS_ARRAY_ND_H_
