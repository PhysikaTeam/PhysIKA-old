/*
 * @file cuda_opengl_vbo_mapper.cpp
 * @Basic class VBOCudaMapper
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

#include <GL/glew.h>
#include <cuda_runtime_api.h>
#include "driver_types.h"
#include "cuda_gl_interop.h"

#include "Physika_Core/Utilities/cuda_utilities.h"

#include "vbo_cuda_mapper.h"

namespace Physika{
    
VBOCudaMapper::VBOCudaMapper(unsigned int VBO)
{
    cudaCheck(cudaGraphicsGLRegisterBuffer(&cuda_graphics_resource_, VBO, cudaGraphicsMapFlagsWriteDiscard));
}

VBOCudaMapper::VBOCudaMapper(unsigned int VBO, unsigned int byte_size)
{
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, byte_size, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaCheck(cudaGraphicsGLRegisterBuffer(&cuda_graphics_resource_, VBO, cudaGraphicsMapFlagsWriteDiscard));
}

VBOCudaMapper::~VBOCudaMapper()
{
    cudaCheck(cudaGraphicsUnregisterResource(cuda_graphics_resource_));
}

std::pair<float *, unsigned int> VBOCudaMapper::mapVBOBuffer()
{
    cudaCheck(cudaGraphicsMapResources(1, &cuda_graphics_resource_, nullptr));
    
    float * cuda_data_ptr = nullptr;
    size_t byte_size;
    cudaCheck(cudaGraphicsResourceGetMappedPointer((void **)&cuda_data_ptr, &byte_size, cuda_graphics_resource_));
    
    return{ cuda_data_ptr, byte_size };
}

void VBOCudaMapper::unmapVBOBuffer()
{
    cudaCheck(cudaGraphicsUnmapResources(1, &cuda_graphics_resource_, nullptr));
}

}//end of namespace Physika