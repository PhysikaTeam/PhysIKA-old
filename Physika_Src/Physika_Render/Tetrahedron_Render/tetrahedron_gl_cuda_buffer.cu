/*
 * @file tetrahedron_gl_cuda_buffer.cu
 * @Basic class CudaTetrahedronGLBuffer
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

#include "tetrahedron_gl_cuda_buffer.h"

namespace Physika{

CPU_GPU_FUNC_DECL TetrahedronGLCudaBuffer::TetrahedronGLCudaBuffer(TriangleGLCudaBuffer cuda_triangle_gl_buffer)
    :triangle_gl_cuda_buffer_(cuda_triangle_gl_buffer)
{
    
}

CPU_GPU_FUNC_DECL unsigned int TetrahedronGLCudaBuffer::tetrahedrondNum() const
{
    return triangle_gl_cuda_buffer_.triangleNum() / 4;
}

template <typename Scalar>
GPU_FUNC_DECL bool TetrahedronGLCudaBuffer::setTetrahedron(unsigned int idx, const Vector<Scalar, 3>& v0, const Vector<Scalar, 3>& v1, const Vector<Scalar, 3>& v2, const Vector<Scalar, 3>& v3, bool auto_compute_normal)
{
    if (idx >= tetrahedrondNum())
        return false;

    triangle_gl_cuda_buffer_.setTriangle(4 * idx + 0, v0, v2, v1, auto_compute_normal);
    triangle_gl_cuda_buffer_.setTriangle(4 * idx + 1, v0, v1, v3, auto_compute_normal);
    triangle_gl_cuda_buffer_.setTriangle(4 * idx + 2, v1, v2, v3, auto_compute_normal);
    triangle_gl_cuda_buffer_.setTriangle(4 * idx + 3, v0, v3, v2, auto_compute_normal);

    return true;
}

template <typename Scalar>
GPU_FUNC_DECL bool TetrahedronGLCudaBuffer::setTetrahedron(unsigned int idx, const Vector<Scalar, 3>& v0, const Vector<Scalar, 3>& v1, const Vector<Scalar, 3>& v2, const Vector<Scalar, 3>& v3, const Vector<Scalar, 3>& face_normal_0, const Vector<Scalar, 3>& face_normal_1, const Vector<Scalar, 3>& face_normal_2, const Vector<Scalar, 3>& face_normal_3)
{
    if (idx >= tetrahedrondNum())
        return false;

    triangle_gl_cuda_buffer_.setTriangle(4 * idx + 0, v0, v2, v1, face_normal_0);
    triangle_gl_cuda_buffer_.setTriangle(4 * idx + 1, v0, v1, v3, face_normal_1);
    triangle_gl_cuda_buffer_.setTriangle(4 * idx + 2, v1, v2, v3, face_normal_2);
    triangle_gl_cuda_buffer_.setTriangle(4 * idx + 3, v0, v3, v2, face_normal_3);

    return true;
}

//explicit instantiations
template GPU_FUNC_DECL bool TetrahedronGLCudaBuffer::setTetrahedron<float>(unsigned int idx, const Vector<float, 3> & v0, const Vector<float, 3> & v1, const Vector<float, 3> & v2, const Vector<float, 3> & v3, bool auto_compute_normal);
template GPU_FUNC_DECL bool TetrahedronGLCudaBuffer::setTetrahedron<double>(unsigned int idx, const Vector<double, 3> & v0, const Vector<double, 3> & v1, const Vector<double, 3> & v2, const Vector<double, 3> & v3, bool auto_compute_normal);

template GPU_FUNC_DECL bool TetrahedronGLCudaBuffer::setTetrahedron<float>(unsigned int idx, const Vector<float, 3> & v0, const Vector<float, 3> & v1, const Vector<float, 3> & v2, const Vector<float, 3> & v3, const Vector<float, 3> & face_normal_0, const Vector<float, 3> & face_normal_1, const Vector<float, 3> & face_normal_2, const Vector<float, 3> & face_normal_3);
template GPU_FUNC_DECL bool TetrahedronGLCudaBuffer::setTetrahedron<double>(unsigned int idx, const Vector<double, 3> & v0, const Vector<double, 3> & v1, const Vector<double, 3> & v2, const Vector<double, 3> & v3, const Vector<double, 3> & face_normal_0, const Vector<double, 3> & face_normal_1, const Vector<double, 3> & face_normal_2, const Vector<double, 3> & face_normal_3);

    
}//end of namespace Physika