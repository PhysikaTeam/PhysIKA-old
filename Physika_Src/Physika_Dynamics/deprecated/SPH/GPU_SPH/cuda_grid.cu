/*
 * @file cuda_grid.cpp 
 * @Brief class CudaGrid
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

#include <algorithm>

#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Dynamics/SPH/GPU_SPH/cuda_grid.h"

namespace Physika{

template<typename T>
CudaGrid<T>::CudaGrid() 
    :nx(0), ny(0), nz(0), nxy(0), data(NULL) 
{

};

//Note: should not release data here.
template<typename T>
CudaGrid<T>::~CudaGrid()
{

};

template<typename T>
__device__ T CudaGrid<T>::operator () (const int i, const int j, const int k) const
{
    return data[i + j*nx + k*nxy];
}

template<typename T>
__device__ T& CudaGrid<T>::operator () (const int i, const int j, const int k)
{
    return data[i + j*nx + k*nxy];
}

template<typename T>
__device__ int CudaGrid<T>::index(const int i, const int j, const int k)
{
    return i + j*nx + k*nxy;
}

template<typename T>
__device__ T CudaGrid<T>::operator [] (const int id) const
{
    return data[id];
}

template<typename T>
__device__ T& CudaGrid<T>::operator [] (const int id)
{
    return data[id];
}


// only swap the pointer
template<typename T>
void CudaGrid<T>::swap(CudaGrid<T> & rhs)
{
    assert(this->nx == rhs.nx && this->ny == rhs.ny && this->nz == rhs.nz);
    std::swap(*this, rhs);
}

template<typename T>
void CudaGrid<T>::setSpace(int nx, int ny, int nz)
{
    this->nx = nx;	
    this->ny = ny;
    this->nz = nz;
    this->nxy = nx*ny;
    this->element_count= nxy*nz;

    this->allocate();
    this->clear();
}

template<typename T>
void CudaGrid<T>::clear()
{
    cudaCheck(cudaMemset(data, 0, element_count * sizeof(T)));
}

template<typename T>
void CudaGrid<T>::allocate()
{
    cudaCheck(cudaMalloc(&data, element_count * sizeof(T)));
}

template<typename T>
void CudaGrid<T>::release()
{
    cudaCheck(cudaFree(data));
}

//explicit instantiation
template class CudaGrid<float>;
template class CudaGrid<float3>;
template class CudaGrid<bool>;

}//end of namespace Physika