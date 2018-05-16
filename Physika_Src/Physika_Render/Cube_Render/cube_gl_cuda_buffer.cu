/*
 * @file cube_gl_cuda_buffer.cu 
 * @Basic class CubeGLCudaBuffer
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

#include "cube_gl_cuda_buffer.h"

namespace Physika{
    
CPU_GPU_FUNC_DECL CubeGLCudaBuffer::CubeGLCudaBuffer(QuadGLCudaBuffer cuda_triangle_gl_buffer)
    :quad_gl_cuda_buffer_(cuda_triangle_gl_buffer)
{
    
}

CPU_GPU_FUNC_DECL unsigned int CubeGLCudaBuffer::cubeNum() const
{
    return quad_gl_cuda_buffer_.quadNum() / 6;
}

template<typename Scalar>
GPU_FUNC_DECL bool CubeGLCudaBuffer::setCube(unsigned int idx, 
                                             const Vector<Scalar, 3> & v0, 
                                             const Vector<Scalar, 3> & v1, 
                                             const Vector<Scalar, 3> & v2, 
                                             const Vector<Scalar, 3> & v3, 
                                             const Vector<Scalar, 3> & v4, 
                                             const Vector<Scalar, 3> & v5,
                                             const Vector<Scalar, 3> & v6,
                                             const Vector<Scalar, 3> & v7,
                                             bool auto_compute_normal)
{
    if (idx >= cubeNum())
        return false;

    quad_gl_cuda_buffer_.setQuad(6 * idx + 0, v0, v3, v2, v1, auto_compute_normal);
    quad_gl_cuda_buffer_.setQuad(6 * idx + 1, v4, v5, v6, v7, auto_compute_normal);
    quad_gl_cuda_buffer_.setQuad(6 * idx + 2, v0, v1, v5, v4, auto_compute_normal);
    quad_gl_cuda_buffer_.setQuad(6 * idx + 3, v3, v7, v6, v2, auto_compute_normal);
    quad_gl_cuda_buffer_.setQuad(6 * idx + 4, v4, v7, v3, v0, auto_compute_normal);
    quad_gl_cuda_buffer_.setQuad(6 * idx + 5, v1, v2, v6, v5, auto_compute_normal);
        

    return true;
}

template<typename Scalar>
GPU_FUNC_DECL bool CubeGLCudaBuffer::setCube(unsigned int idx, 
                                             const Vector<Scalar, 3> & v0, 
                                             const Vector<Scalar, 3> & v1, 
                                             const Vector<Scalar, 3> & v2, 
                                             const Vector<Scalar, 3> & v3, 
                                             const Vector<Scalar, 3> & v4, 
                                             const Vector<Scalar, 3> & v5,
                                             const Vector<Scalar, 3> & v6,
                                             const Vector<Scalar, 3> & v7,
                                             const Vector<Scalar, 3> & face_normal_0,
                                             const Vector<Scalar, 3> & face_normal_1,
                                             const Vector<Scalar, 3> & face_normal_2,
                                             const Vector<Scalar, 3> & face_normal_3, 
                                             const Vector<Scalar, 3> & face_normal_4,
                                             const Vector<Scalar, 3> & face_normal_5)
{
    if (idx >= cubeNum())
        return false;

    quad_gl_cuda_buffer_.setQuad(6 * idx + 0, v0, v3, v2, v1, face_normal_0);
    quad_gl_cuda_buffer_.setQuad(6 * idx + 1, v4, v5, v6, v7, face_normal_1);
    quad_gl_cuda_buffer_.setQuad(6 * idx + 2, v0, v1, v5, v4, face_normal_2);
    quad_gl_cuda_buffer_.setQuad(6 * idx + 3, v3, v7, v6, v2, face_normal_3);
    quad_gl_cuda_buffer_.setQuad(6 * idx + 4, v4, v7, v3, v0, face_normal_4);
    quad_gl_cuda_buffer_.setQuad(6 * idx + 5, v1, v2, v6, v5, face_normal_5);
        

    return true;
}

//explicit instantiations
template GPU_FUNC_DECL bool CubeGLCudaBuffer::setCube<float>(unsigned int idx, const Vector<float, 3> & v0, const Vector<float, 3> & v1, const Vector<float, 3> & v2, const Vector<float, 3> & v3, const Vector<float, 3> & v4, const Vector<float, 3> & v5, const Vector<float, 3> & v6, const Vector<float, 3> & v7, bool auto_compute_normal);
template GPU_FUNC_DECL bool CubeGLCudaBuffer::setCube<double>(unsigned int idx, const Vector<double, 3> & v0, const Vector<double, 3> & v1, const Vector<double, 3> & v2, const Vector<double, 3> & v3, const Vector<double, 3> & v4, const Vector<double, 3> & v5, const Vector<double, 3> & v6, const Vector<double, 3> & v7, bool auto_compute_normal);

template GPU_FUNC_DECL bool CubeGLCudaBuffer::setCube<float>(unsigned int idx, const Vector<float, 3> & v0, const Vector<float, 3> & v1, const Vector<float, 3> & v2, const Vector<float, 3> & v3, const Vector<float, 3> & v4, const Vector<float, 3> & v5, const Vector<float, 3> & v6, const Vector<float, 3> & v7, const Vector<float, 3> & face_normal_0, const Vector<float, 3> & face_normal_1, const Vector<float, 3> & face_normal_2, const Vector<float, 3> & face_normal_3, const Vector<float, 3> & face_normal_4, const Vector<float, 3> & face_normal_5);
template GPU_FUNC_DECL bool CubeGLCudaBuffer::setCube<double>(unsigned int idx, const Vector<double, 3> & v0, const Vector<double, 3> & v1, const Vector<double, 3> & v2, const Vector<double, 3> & v3, const Vector<double, 3> & v4, const Vector<double, 3> & v5, const Vector<double, 3> & v6, const Vector<double, 3> & v7, const Vector<double, 3> & face_normal_0, const Vector<double, 3> & face_normal_1, const Vector<double, 3> & face_normal_2, const Vector<double, 3> & face_normal_3, const Vector<double, 3> & face_normal_4, const Vector<double, 3> & face_normal_5);


}//end of namespace Physika