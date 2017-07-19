/*
 * @file boundary.h
 * @Brief boundary condition
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


#ifndef PHYSIKA_DYNAMICS_SPH_GPU_SPH_BOUNDARY_H_
#define PHYSIKA_DYNAMICS_SPH_GPU_SPH_BOUNDARY_H_

#include <vector>
#include "vector_types.h"
#include "Physika_Dynamics/SPH/GPU_SPH/cuda_distance_field.h"

namespace Physika{

template <typename ElementType> class CudaArray;

class Barrier
{
public:
    Barrier() = default;
    ~Barrier() = default;

    virtual void constrain(CudaArray<float3>& pos_arr, CudaArray<float3>& vel_arr, float dt) const = 0;

    void setNormalFriction(float normal_friction);
    void setTangentialFriction(float tangential_friction);

protected:
    float normal_friction_ = 0.95f;
    float tangential_friction_ = 0.0f;
};

//=======================================================================================================================

class BarrierCudaDistanceField : public Barrier 
{

public:

    //Note: normal n should point outwards, i.e., away from inside of constraint
    BarrierCudaDistanceField(CudaDistanceField *df);
    ~BarrierCudaDistanceField();

    virtual void constrain(CudaArray<float3>& pos_arr, CudaArray<float3>& vel_arr, float dt) const;

private:
    CudaDistanceField * distance_field_;
};

//=======================================================================================================================

class Boundary
{
public:
    Boundary();
    ~Boundary();

    void constrain(CudaArray<float3>& pos_arr, CudaArray<float3>& vel_arr, float dt);

    void insertBarrier(Barrier *barrier);
    unsigned int size() const;

public:
    std::vector<Barrier *> barriers_;
};

}//end of namespace Physika

#endif //PHYSIKA_DYNAMICS_SPH_GPU_SPH_BOUNDARY_H_