/*
 * @file cuda_distance_field.h 
 * @Brief class CudaDistanceField
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


#ifndef PHYSIKA_DYNAMICS_SPH_GPU_SPH_CUDA_DISTANCE_FIELD_H_
#define PHYSIKA_DYNAMICS_SPH_GPU_SPH_CUDA_DISTANCE_FIELD_H_

#include <string>
#include "vector_types.h"
#include "vector_functions.h"

#include "Physika_Dynamics/SPH/GPU_SPH/cuda_grid.h"

namespace Physika{


class CudaDistanceField 
{

public:
    CudaDistanceField();
    CudaDistanceField(std::string filename);

    // Note: should not release data here, call release() explicitly.
    ~CudaDistanceField();

    void setSpace(const float3 p0, const float3 p1, int nbx, int nby, int nbz);
    void release();

    void translate(const float3 & t);
    void scale(const float s);
    void invert();

    void distanceFieldToBox(float3& lo, float3& hi, bool inverted);
    void distanceFieldToCylinder(float3& center, float radius, float height, int axis, bool inverted);
    void distanceFieldToSphere(float3& center, float radius, bool inverted);

    __device__ void getDistance(const float3 &p, float &d, float3 &g);
    __host__ __device__  float lerp(float a, float b, float alpha) const;

public:
    void readSDF(std::string filename);

public:
    float3 left;		// lower left front corner
    float3 h;			// single cell sizes

    CudaGrid1f gDist;
    bool bInvert;
};



}//end of namespace Physika

#endif //PHYSIKA_DYNAMICS_SPH_GPU_SPH_CUDA_DISTANCE_FIELD_H_