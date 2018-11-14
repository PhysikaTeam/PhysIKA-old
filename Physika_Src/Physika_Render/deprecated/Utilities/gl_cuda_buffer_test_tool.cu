/*
 * @file gl_cuda_buffer_tool.cu 
 * @Basic Test tool for ***GLCudaBuffer
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

#include <cmath>
#include "cuda_runtime.h"

#include "Physika_Render/Point_Render/point_gl_cuda_buffer.h"
#include "Physika_Render/Line_Render/line_gl_cuda_buffer.h"
#include "Physika_Render/Triangle_Render/triangle_gl_cuda_buffer.h"

#include "Physika_Render/Tetrahedron_Render/tetrahedron_gl_cuda_buffer.h"
#include "Physika_Render/Quad_Render/quad_gl_cuda_buffer.h"
#include "Physika_Render/Cube_Render/cube_gl_cuda_buffer.h"

#include "gl_cuda_buffer_test_tool.h"

namespace Physika{
    

__global__ void setPointGLCudaBufferKernel(float * dev_ptr, unsigned int point_num, PointGLCudaBuffer point_gl_cuda_buffer)
{
    unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (thread_idx >= point_num)
        return;

    Vector3f pos = { dev_ptr[3 * thread_idx], dev_ptr[3 * thread_idx + 1], dev_ptr[3 * thread_idx + 2] };
    point_gl_cuda_buffer.setPoint(thread_idx, pos);
}


void setPointGLCudaBuffer(const std::vector<Vector3f> & point_data, PointGLCudaBuffer & point_gl_cuda_buffer)
{
    float * dev_ptr;
    cudaCheck(cudaMalloc(&dev_ptr, sizeof(Vector3f) * point_data.size()));
    cudaCheck(cudaMemcpy(dev_ptr, point_data.data(), sizeof(Vector3f) * point_data.size(), cudaMemcpyHostToDevice));

    unsigned int thread_num = 256;
    unsigned int block_num = std::ceil(static_cast<float>(point_data.size()) / thread_num);

    setPointGLCudaBufferKernel<<<block_num, thread_num >>> (dev_ptr, point_data.size(), point_gl_cuda_buffer);

    cudaDeviceSynchronize();
    cudaFree(dev_ptr);
}


__global__ void setLineGLCudaBufferKernel(float * dev_ptr, unsigned int line_num, LineGLCudaBuffer line_gl_cuda_buffer)
{
    unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx >= line_num)
        return;

    Vector3f v0 = { dev_ptr[6 * thread_idx + 0], dev_ptr[6 * thread_idx + 1], dev_ptr[6 * thread_idx + 2] };
    Vector3f v1 = { dev_ptr[6 * thread_idx + 3], dev_ptr[6 * thread_idx + 4], dev_ptr[6 * thread_idx + 5] };

    line_gl_cuda_buffer.setLine(thread_idx, v0, v1);
}


void setLineGLCudaBuffer(const std::vector<Vector3f> & line_data, LineGLCudaBuffer & line_gl_cuda_buffer)
{
    float * dev_ptr;
    cudaCheck(cudaMalloc(&dev_ptr, sizeof(Vector3f) * line_data.size()));
    cudaCheck(cudaMemcpy(dev_ptr, line_data.data(), sizeof(Vector3f) * line_data.size(), cudaMemcpyHostToDevice));

    unsigned int line_num = line_data.size() / 2;
    unsigned int thread_num = 256;
    unsigned int block_num = std::ceil(static_cast<float>(line_num) / thread_num);

    setLineGLCudaBufferKernel << <block_num, thread_num >> > (dev_ptr, line_num, line_gl_cuda_buffer);

    cudaDeviceSynchronize();
    cudaFree(dev_ptr);
}

__global__ void setTriangleGLCudaBufferKernel(float * dev_ptr, unsigned int triangle_num, TriangleGLCudaBuffer triangle_gl_cuda_buffer)
{
    unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx >= triangle_num)
        return;

    Vector3f v0 = { dev_ptr[9 * thread_idx + 0], dev_ptr[9 * thread_idx + 1], dev_ptr[9 * thread_idx + 2] };
    Vector3f v1 = { dev_ptr[9 * thread_idx + 3], dev_ptr[9 * thread_idx + 4], dev_ptr[9 * thread_idx + 5] };
    Vector3f v2 = { dev_ptr[9 * thread_idx + 6], dev_ptr[9 * thread_idx + 7], dev_ptr[9 * thread_idx + 8] };

    triangle_gl_cuda_buffer.setTriangle(thread_idx, v0, v1, v2);
}


void setTriangleGLCudaBuffer(const std::vector<Vector3f> & triangle_data, TriangleGLCudaBuffer & triangle_gl_cuda_buffer)
{
    float * dev_ptr;
    cudaCheck(cudaMalloc(&dev_ptr, sizeof(Vector3f) * triangle_data.size()));
    cudaCheck(cudaMemcpy(dev_ptr, triangle_data.data(), sizeof(Vector3f) * triangle_data.size(), cudaMemcpyHostToDevice));

    unsigned int triangle_num = triangle_data.size() / 3;
    unsigned int thread_num = 256;
    unsigned int block_num = std::ceil(static_cast<float>(triangle_num) / thread_num);

    setTriangleGLCudaBufferKernel << <block_num, thread_num >> > (dev_ptr, triangle_num, triangle_gl_cuda_buffer);

    cudaDeviceSynchronize();
    cudaFree(dev_ptr);
}

__global__ void setTetrahedronGLCudaBufferKernel(float * dev_ptr, unsigned int tet_num, TetrahedronGLCudaBuffer tet_gl_cuda_buffer)
{
    unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx >= tet_num)
        return;

    Vector3f v0 = { dev_ptr[12 * thread_idx + 0], dev_ptr[12 * thread_idx + 1], dev_ptr[12 * thread_idx + 2] };
    Vector3f v1 = { dev_ptr[12 * thread_idx + 3], dev_ptr[12 * thread_idx + 4], dev_ptr[12 * thread_idx + 5] };
    Vector3f v2 = { dev_ptr[12 * thread_idx + 6], dev_ptr[12 * thread_idx + 7], dev_ptr[12 * thread_idx + 8] };
    Vector3f v3 = { dev_ptr[12 * thread_idx + 9], dev_ptr[12 * thread_idx + 10], dev_ptr[12 * thread_idx + 11] };

    tet_gl_cuda_buffer.setTetrahedron(thread_idx, v0, v1, v2, v3);
}


void setTetrahedronGLCudaBuffer(const std::vector<Vector3f> & tet_data, TetrahedronGLCudaBuffer & tet_gl_cuda_buffer)
{
    float * dev_ptr;
    cudaCheck(cudaMalloc(&dev_ptr, sizeof(Vector3f) * tet_data.size()));
    cudaCheck(cudaMemcpy(dev_ptr, tet_data.data(), sizeof(Vector3f) * tet_data.size(), cudaMemcpyHostToDevice));

    unsigned int tet_num = tet_data.size() / 4;
    unsigned int thread_num = 256;
    unsigned int block_num = std::ceil(static_cast<float>(tet_num) / thread_num);

    setTetrahedronGLCudaBufferKernel << <block_num, thread_num >> > (dev_ptr, tet_num, tet_gl_cuda_buffer);

    cudaDeviceSynchronize();
    cudaFree(dev_ptr);
}

__global__ void setQuadGLCudaBufferKernel(float * dev_ptr, unsigned int quad_num, QuadGLCudaBuffer quad_gl_cuda_buffer)
{
    unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx >= quad_num)
        return;

    Vector3f v0 = { dev_ptr[12 * thread_idx + 0], dev_ptr[12 * thread_idx + 1], dev_ptr[12 * thread_idx + 2] };
    Vector3f v1 = { dev_ptr[12 * thread_idx + 3], dev_ptr[12 * thread_idx + 4], dev_ptr[12 * thread_idx + 5] };
    Vector3f v2 = { dev_ptr[12 * thread_idx + 6], dev_ptr[12 * thread_idx + 7], dev_ptr[12 * thread_idx + 8] };
    Vector3f v3 = { dev_ptr[12 * thread_idx + 9], dev_ptr[12 * thread_idx + 10], dev_ptr[12 * thread_idx + 11] };

    quad_gl_cuda_buffer.setQuad(thread_idx, v0, v1, v2, v3);
}


void setQuadGLCudaBuffer(const std::vector<Vector3f> & quad_data, QuadGLCudaBuffer & quad_gl_cuda_buffer)
{
    float * dev_ptr;
    cudaCheck(cudaMalloc(&dev_ptr, sizeof(Vector3f) * quad_data.size()));
    cudaCheck(cudaMemcpy(dev_ptr, quad_data.data(), sizeof(Vector3f) * quad_data.size(), cudaMemcpyHostToDevice));

    unsigned int quad_num = quad_data.size() / 4;
    unsigned int thread_num = 256;
    unsigned int block_num = std::ceil(static_cast<float>(quad_num) / thread_num);

    setQuadGLCudaBufferKernel << <block_num, thread_num >> > (dev_ptr, quad_data.size(), quad_gl_cuda_buffer);

    cudaDeviceSynchronize();
    cudaFree(dev_ptr);
}

__global__ void setCubeGLCudaBufferKernel(float * dev_ptr, unsigned int cube_num, CubeGLCudaBuffer cube_gl_cuda_buffer)
{
    unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx >= cube_num)
        return;

    Vector3f v0 = { dev_ptr[24 * thread_idx + 0], dev_ptr[24 * thread_idx + 1], dev_ptr[24 * thread_idx + 2] };
    Vector3f v1 = { dev_ptr[24 * thread_idx + 3], dev_ptr[24 * thread_idx + 4], dev_ptr[24 * thread_idx + 5] };
    Vector3f v2 = { dev_ptr[24 * thread_idx + 6], dev_ptr[24 * thread_idx + 7], dev_ptr[24 * thread_idx + 8] };
    Vector3f v3 = { dev_ptr[24 * thread_idx + 9], dev_ptr[24 * thread_idx + 10], dev_ptr[24 * thread_idx + 11] };
    Vector3f v4 = { dev_ptr[24 * thread_idx + 12], dev_ptr[24 * thread_idx + 13], dev_ptr[24 * thread_idx + 14] };
    Vector3f v5 = { dev_ptr[24 * thread_idx + 15], dev_ptr[24 * thread_idx + 16], dev_ptr[24 * thread_idx + 17] };
    Vector3f v6 = { dev_ptr[24 * thread_idx + 18], dev_ptr[24 * thread_idx + 19], dev_ptr[24 * thread_idx + 20] };
    Vector3f v7 = { dev_ptr[24 * thread_idx + 21], dev_ptr[24 * thread_idx + 22], dev_ptr[24 * thread_idx + 23] };

    cube_gl_cuda_buffer.setCube(thread_idx, v0, v1, v2, v3, v4, v5, v6, v7);
}


void setCubeGLCudaBuffer(const std::vector<Vector3f> & cube_data, CubeGLCudaBuffer & cube_gl_cuda_buffer)
{
    float * dev_ptr;
    cudaCheck(cudaMalloc(&dev_ptr, sizeof(Vector3f) * cube_data.size()));
    cudaCheck(cudaMemcpy(dev_ptr, cube_data.data(), sizeof(Vector3f) * cube_data.size(), cudaMemcpyHostToDevice));

    unsigned int cube_num = cube_data.size() / 8;
    unsigned int thread_num = 256;
    unsigned int block_num = std::ceil(static_cast<float>(cube_num) / thread_num);

    setCubeGLCudaBufferKernel << <block_num, thread_num >> > (dev_ptr, cube_num, cube_gl_cuda_buffer);

    cudaDeviceSynchronize();
    cudaFree(dev_ptr);
}



}//end of namespace Physika