/*
 * @file cuda_pbd_sph.cpp
 * @Brief class CudaPBDSPH
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


#include <sstream>
#include <iostream>
#include <fstream>
#include <string>

#include "Physika_Core/Timer/timer.h"
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Dynamics/SPH/GPU_SPH/cuda_pbd_sph.h"


namespace Physika{


CudaPBDSPH::CudaPBDSPH(Settings settings)
    :params_(settings), sim_itor_(0)
{

}

CudaPBDSPH::~CudaPBDSPH()
{
    grid_hash_.release();
}

float CudaPBDSPH::getTimeStep()
{
    return 0.008f;
}

unsigned int CudaPBDSPH::numFluidParticle() const
{
    return this->pos_arr_.size();
}

void CudaPBDSPH::advanceStep(float dt)
{
    //==============================================================================
    //plugin operation, begin time step
    SPHPluginBase<float, 3> * plugin = NULL;
    for (unsigned int i = 0; i < this->plugins_.size(); i++)
    {
        plugin = dynamic_cast<SPHPluginBase<float, 3>*>(this->plugins_[i]);
        if (plugin)
            plugin->onBeginTimeStep(this->time_, dt);
    }
    //==============================================================================

    this->step(this->getTimeStep());
    ++sim_itor_;

    //==============================================================================
    //plugin operation, end time step
    for (unsigned int i = 0; i < this->plugins_.size(); i++)
    {
        plugin = dynamic_cast<SPHPluginBase<float, 3> *>(this->plugins_[i]);
        plugin->onEndTimeStep(this->time_, dt);
    }
    //==============================================================================

}

void CudaPBDSPH::allocateMemory()
{
    std::cout << "Total particle number: " << params_.pNum << std::endl;

    pos_arr_.resize(params_.pNum);
    vel_arr_.resize(params_.pNum);
    rho_arr_.resize(params_.pNum);

    lambda_arr_.resize(params_.pNum);

    buffer_.resize(params_.pNum);

    old_vel_.resize(params_.pNum);

    neighbors_arr_.resize(params_.pNum);

    //data = new Vector3f[params.pNum];
    host_copy_ = new float4[params_.pNum];
}

void CudaPBDSPH::computeNeighbors()
{
    grid_hash_.queryNeighbors(pos_arr_, neighbors_arr_, params_.smoothingLength, params_.samplingDistance, NEIGHBOR_SIZE);
    //hash.QueryNeighborSlow(pos_arr_, neighborsArr, params.smoothingLength, NEIGHBOR_SIZE);
}

void CudaPBDSPH::boundaryHandling(float dt)
{
    boundary_.constrain(pos_arr_, vel_arr_, dt);

    //uint pDims = int(ceil(pos_arr_.Size() / BLOCK_SIZE + 0.5f));
    //K_ConstrainPosition <<< pDims, BLOCK_SIZE >>>(pos_arr_, velArr, hash.lo+0.02f, hash.hi-0.02f);
}

void CudaPBDSPH::correctWithPBD(float dt)
{
    int total_itoration = 3;

    int itor = 0;
    do
    {
        computeDensity();
        iterationPBD(dt);
        itor++;
    } while (itor <= total_itoration);
}

void CudaPBDSPH::addModel(Model* model) 
{ 
    models_.push_back(model); 
}

CudaArray<float3> CudaPBDSPH::getPosArr()
{
    return this->pos_arr_;
}

}//end of Physika