/*
 * @file cuda_pbd_sph.h
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


#ifndef PHYSIKA_DYNAMICS_SPH_GPU_SPH_CUDA_PBD_SPH_H_
#define PHYSIKA_DYNAMICS_SPH_GPU_SPH_CUDA_PBD_SPH_H_

#include <vector>
#include "vector_types.h"

#include "Physika_Core/Cuda_Array/cuda_array.h"
#include "Physika_Dynamics/SPH/GPU_SPH/boundary.h"
#include "Physika_Dynamics/SPH/GPU_SPH/cuda_grid_hash.h"
#include "Physika_Dynamics/SPH/GPU_SPH/model.h"

#include "Physika_Dynamics/SPH/sph_base.h"
#include "Physika_Dynamics/SPH/sph_plugin_base.h"

namespace Physika{

struct Settings 
{
    int pNum;
    int nbrMaxSize;

    float mass;

    float3 lowBound;
    float3 upBound;

    float3 gravity;

    int iterNum;

    float smoothingLength;
    float samplingDistance;

    float restDensity;
    float maxAii;

    float surfaceTension;
    float viscosity;

    float normalFriction;
    float tangentialFriction;

    float elipsedTime;
};

class CudaPBDSPH: public SPHBase<float, 3>
{
public:

    CudaPBDSPH(Settings settings);
    ~CudaPBDSPH();

    unsigned int numFluidParticle() const;

    virtual float getTimeStep();

    virtual void advanceStep(float dt);

    void allocateMemory();

    void computeNeighbors();
    void step(float dt);

    void predict(float dt);
    void computeDensity();

    void correctWithPBD(float dt);
    void iterationPBD(float dt);

    void applyViscosity(float dt);
    void boundaryHandling(float dt);

    CudaArray<float3> getPosArr();

public:
    void addModel(Model* model);

    virtual bool initialize(std::string in_filename);
    void initialSceneBoundary();

private:
    int sim_itor_;

    CudaArray<float3> pos_arr_;
    CudaArray<float3> vel_arr_;

    CudaArray<float> rho_arr_;
    CudaArray<float> lambda_arr_;

    CudaArray<float3> buffer_;

    CudaArray<float3> old_vel_;


    CudaArray<NeighborList> neighbors_arr_;

    CudaGridHash grid_hash_;

    Settings params_;

    float4* host_copy_;

    Boundary boundary_;

    std::vector<Model*> models_;

};



}//end of namespace Physika

#endif //PHYSIKA_DYNAMICS_SPH_GPU_SPH_CUDA_PBD_SPH_H_