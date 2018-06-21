/*
 * @file cuda_array.h
 * @Brief class CudaArray, Array allocated in GPU
 * @author Wei Chen
 *
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0.
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_CUDA_ARRAY_CUDA_ARRAY_H_
#define PHYSIKA_CORE_CUDA_ARRAY_CUDA_ARRAY_H_

#include <vector>
#include <cuda_runtime.h>

#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Core/Utilities/physika_assert.h"

namespace Physika {

//Note: ElementType objects are allocated in GPU memory

template<typename ElementType>
class CudaArray
{
public:
    CudaArray();
    CudaArray(unsigned int element_count);

    ~CudaArray(); //warning: cuda memory will not free here

    void resize(unsigned int element_count);
    void release(); //note: user must release memory manually 

    //rest all bits to 0
    void reset(); 

    void copyFromHost(const std::vector<ElementType> & host_data);
    void swap(CudaArray & rhs);

    __host__ __device__ ElementType * data();
    __host__ __device__ unsigned int size() const;

    __device__ ElementType & operator [] (unsigned int id);
    __device__ const ElementType & operator [](unsigned int id) const;

protected:
    void allocate();
    
private:
    ElementType * data_;
    unsigned int element_count_;
};

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename ElementType>
CudaArray<ElementType>::CudaArray()
    :data_(nullptr), element_count_(0)
{

}

template<typename ElementType>
CudaArray<ElementType>::CudaArray(unsigned int element_count)
    :data_(nullptr)
{
    this->resize(element_count);
}

template<typename ElementType>
CudaArray<ElementType>::~CudaArray()
{

}

template<typename ElementType>
void CudaArray<ElementType>::resize(unsigned int element_count)
{
    if (this->data_ != nullptr)
        this->release();

    this->element_count_ = element_count;

    this->allocate();
    this->reset();
}

template<typename ElementType>
void CudaArray<ElementType>::reset()
{
    cudaCheck(cudaMemset(data_, 0, element_count_ * sizeof(ElementType)));
}

template<typename ElementType>
void CudaArray<ElementType>::copyFromHost(const std::vector<ElementType> & host_data)
{
    PHYSIKA_ASSERT(this->size() == host_data.size());
    cudaCheck(cudaMemcpy(this->data_, host_data.data(), this->element_count_*sizeof(ElementType), cudaMemcpyHostToDevice));
}

template<typename ElementType>
void CudaArray<ElementType>::swap(CudaArray<ElementType> & rhs)
{
    std::swap(this->data_, rhs.data_);
    std::swap(this->element_count_, rhs.element_count_);
}

template<typename ElementType>
__host__ __device__ ElementType * CudaArray<ElementType>::data()
{
    return this->data_;
}

template<typename ElementType>
__host__ __device__ unsigned int CudaArray<ElementType>::size() const
{
    return this->element_count_;
}

template<typename ElementType>
__device__ ElementType & CudaArray<ElementType>::operator[](unsigned int id)
{
    return this->data_[id];
}

template<typename ElementType>
__device__ const ElementType & CudaArray<ElementType>::operator[](unsigned int id) const
{
    return this->data_[id];
}

template<typename ElementType>
void CudaArray<ElementType>::allocate()
{
    cudaCheck(cudaMalloc((void**)&this->data_, this->element_count_*sizeof(ElementType)));
}

template<typename ElementType>
void CudaArray<ElementType>::release()
{
    cudaCheck(cudaFree(this->data_));
}


}//end of namespace Physika

#endif //PHYSIKA_CORE_CUDA_ARRAY_CUDA_ARRAY_H_