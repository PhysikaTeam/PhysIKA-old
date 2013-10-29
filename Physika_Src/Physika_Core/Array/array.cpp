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

#include "Physika_Core/Array/array.h"

namespace Physika{
template <typename Scalar>
Array<Scalar>::Array():element_cout_(0)
{
    data_ = NULL;
}

template <typename Scalar>
Array<Scalar>::Array(Scalar* data, unsigned int element_cout)
{
    setSpace(element_cout);
    memcpy(data_,data,sizeof(Scalar)*element_cout_);
}

template <typename Scalar>
Array<Scalar>::Array(const Array<Scalar>& arr)
{
    setSpace(arr.element_cout());
    memcpy(data_,arr.data(),sizeof(Scalar)*element_cout_);
}

template <typename Scalar>
Array<Scalar>::~Array()
{
    release();
}

template <typename Scalar>
void Array<Scalar>::allocate()
{
    data_ = new Scalar[element_cout_];
}

template <typename Scalar>
void Array<Scalar>::release()
{
    if(data_ != NULL)
        delete [] data_;
    data_ = NULL;
}

template <typename Scalar>
void Array<Scalar>::reset(const unsigned int count)
{
    release();
    setSpace(count);
}

template <typename Scalar>
void Array<Scalar>::setSpace(const unsigned int count)
{
    element_cout_ = count;
    allocate();
}

template <typename Scalar>
void Array<Scalar>::zero()
{
    memset((void*)data_, 0, element_cout_*sizeof(Scalar));
}


template <typename Scalar>
Array<Scalar>& Array<Scalar>::operator = (const Array<Scalar>& arr)
{
    reset(arr.element_cout());
    memcpy(data_,arr.data(),sizeof(Scalar)*element_cout_);
    return *this;
}

//convenient typedefs
template class Array<int> ;
template class Array<float> ;
template class Array<double> ;
template class Array<Vector3f> ;
template class Array<Vector3d> ;

}