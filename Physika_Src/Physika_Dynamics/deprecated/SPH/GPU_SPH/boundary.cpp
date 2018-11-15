/*
 * @file boundary.cpp
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

#include "Physika_Dynamics/SPH/GPU_SPH/boundary.h"

namespace Physika{

void Barrier::setNormalFriction(float normal_friction)
{
    this->normal_friction_ = normal_friction;
}

void Barrier::setTangentialFriction(float tangential_friction)
{
    this->tangential_friction_ = tangential_friction;
}

//Note: normal n should point outwards, i.e., away from inside of constraint
BarrierCudaDistanceField::BarrierCudaDistanceField(CudaDistanceField *df)
    :Barrier(), distance_field_(df)
{
}

BarrierCudaDistanceField::~BarrierCudaDistanceField()
{
    this->distance_field_->release();
}

Boundary::Boundary()
{
}

Boundary::~Boundary()
{
    for (int i = 0; i < barriers_.size(); i++)
        delete barriers_[i];
    barriers_.clear();
}

void Boundary::constrain(CudaArray<float3>& pos_arr, CudaArray<float3>& vel_arr, float dt)
{
    for (int i = 0; i < barriers_.size(); i++)
        barriers_[i]->constrain(pos_arr, vel_arr, dt);
}

void Boundary::insertBarrier(Barrier *barrier)
{
    barriers_.push_back(barrier);
}

unsigned int Boundary::size() const
{
    return barriers_.size();
}

}//end of namespace Physika