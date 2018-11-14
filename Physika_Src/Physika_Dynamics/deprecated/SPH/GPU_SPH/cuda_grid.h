/*
 * @file cuda_grid.h 
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

#ifndef PHYSIKA_DYNAMICS_SPH_GPU_SPH_GPU_GRID_H_
#define PHYSIKA_DYNAMICS_SPH_GPU_SPH_GPU_GRID_H_

#include "vector_types.h"

namespace Physika{

template<typename T>
class CudaGrid
{
public:
    CudaGrid();
    ~CudaGrid(); //Note: should not release data here.

    __device__ T  operator () (const int i, const int j, const int k) const;
    __device__ T& operator () (const int i, const int j, const int k);
    __device__ T  operator [] (const int id) const;
    __device__ T& operator [] (const int id);
    __device__ int index(const int i, const int j, const int k);

    // only swap the pointer
    void swap(CudaGrid<T>& rhs);
    void setSpace(int nx, int ny, int nz);

    void clear();
    void allocate();
    void release();

public:
    int nx;
    int ny;
    int nz;
    int nxy;
    int element_count;
    T *	data;
};

typedef CudaGrid<float>	 CudaGrid1f;
typedef CudaGrid<float3> CudaGrid3f;
typedef CudaGrid<bool>   CudaGrid1b;


}

#endif //PHYSIKA_DYNAMICS_SPH_GPU_SPH_GPU_GRID_H_