/*
* @file cuda_pbd_sph_plugin_render.cu
* @brief base class CudaPBDSPHPluginRender.
* @author Wei Chen
*
* This file is part of Physika, a versatile physics simulation library.
* Copyright (C) 2013 Physika Group.
*
* This Source Code Form is subject to the terms of the GNU General Public License v2.0.
* If a copy of the GPL was not distributed with this file, you can obtain one at:
* http://www.gnu.org/licenses/gpl-2.0.html
*
*/

#include <iostream>
#include <GL/freeglut.h>
#include "Physika_Dynamics/SPH/GPU_SPH/cuda_pbd_sph.h"
#include "Physika_Dynamics/SPH/GPU_SPH/cuda_pbd_sph_plugin_render.h"

namespace Physika {

/*
Note: It seem that cuda nvcc complier is not compatible with Eigen.
      So we use another .cu file to complier our cuda codes.
*/

__global__ void K_CopyPosRenderData(float4 * dst, CudaArray<float3> src)
{
    unsigned int thread_id = blockIdx.x *  blockDim.x + threadIdx.x;

    if (thread_id >= src.size())
        return;

    dst[thread_id].x = src[thread_id].x;
    dst[thread_id].y = src[thread_id].y;
    dst[thread_id].z = src[thread_id].z;
    dst[thread_id].w = 1;

}


void CudaPBDSPHPluginRender::onEndTimeStep(float time, float dt)
{
    float4 * pos_dst = this->mapVBOBuffer();

    CudaPBDSPH * cuda_pbd_sph = dynamic_cast<CudaPBDSPH *>(active_instance_->driver());
    CudaArray<float3> pos_src = cuda_pbd_sph->getPosArr();

    dim3 pDims = int(ceil(pos_src.size() / BLOCK_SIZE + 0.5f));
    K_CopyPosRenderData<<<pDims, BLOCK_SIZE>>>(pos_dst, pos_src);

    this->unmapVBOBuffer();

    ///////////////////////////////////////////////////////////////////////////////////////

    /*
    float * pos_copy = new float[pos_src.size() * 3];
    cudaCheck(cudaMemcpy(pos_copy, pos_src.data(), pos_src.size() * sizeof(float) * 3, cudaMemcpyDeviceToHost));
    for (unsigned int id = 0; id < pos_src.size(); ++id)
        std::cout << "id: " << pos_copy[3 * id] << pos_copy[3 * id + 1] << pos_copy[3 * id + 2] << std::endl;
    */

    //////////////////////////////////////////////////////////////////////////////////////

    glutMainLoopEvent();
    glutPostRedisplay();
}

}//end of namespace Physika