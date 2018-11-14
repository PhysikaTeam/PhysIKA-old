/*
 * @file cuda_opengl_vbo_mapper.h 
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

#pragma once

#include <utility>

struct cudaGraphicsResource;

namespace Physika{

/*
 * class VBOCudaMapper is used to map OpenGL VBO graphic resources to cuda memory.
 */
    
class VBOCudaMapper
{
public:
    VBOCudaMapper(unsigned int VBO);
    VBOCudaMapper(unsigned int VBO, unsigned int byte_size);

    VBOCudaMapper(const VBOCudaMapper &) = delete;
    VBOCudaMapper & operator = (const VBOCudaMapper &) = delete;

    ~VBOCudaMapper();

    std::pair<float *, unsigned int> mapVBOBuffer(); //return <device_ptr, byte_size>
    void unmapVBOBuffer();

private:
    cudaGraphicsResource * cuda_graphics_resource_;
};

}//end of namespace Physika