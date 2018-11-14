/*
 * @file quad_gl_cuda_buffer.cu
 * @Basic class CudaQuadGLBuffer
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

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"

#include "quad_gl_cuda_buffer.h"

namespace Physika {

COMM_FUNC QuadGLCudaBuffer::QuadGLCudaBuffer(LineGLCudaBuffer cuda_line_gl_buffer, TriangleGLCudaBuffer cuda_triangle_gl_buffer)
    :line_gl_cuda_buffer_(cuda_line_gl_buffer), triangle_gl_cuda_buffer_(cuda_triangle_gl_buffer)
{
    
}

COMM_FUNC unsigned int QuadGLCudaBuffer::quadNum() const
{
//    PHYSIKA_ASSERT(line_gl_cuda_buffer_.lineNum() / 4 == triangle_gl_cuda_buffer_.triangleNum() / 2);
    return triangle_gl_cuda_buffer_.triangleNum() / 2;
}

template<typename Scalar, int Dim>
GPU_FUNC bool QuadGLCudaBuffer::setQuad(unsigned int idx, 
                                             const Vector<Scalar, Dim>& v0, 
                                             const Vector<Scalar, Dim>& v1, 
                                             const Vector<Scalar, Dim>& v2, 
                                             const Vector<Scalar, Dim>& v3, 
                                             bool auto_compute_normal)
{
    if (idx >= quadNum())
        return false;

    line_gl_cuda_buffer_.setLine(4 * idx + 0, v0, v1);
    line_gl_cuda_buffer_.setLine(4 * idx + 1, v1, v2);
    line_gl_cuda_buffer_.setLine(4 * idx + 2, v2, v3);
    line_gl_cuda_buffer_.setLine(4 * idx + 3, v3, v0);

    triangle_gl_cuda_buffer_.setTriangle(2 * idx + 0, v0, v1, v2, auto_compute_normal);
    triangle_gl_cuda_buffer_.setTriangle(2 * idx + 1, v0, v2, v3, auto_compute_normal);


    return true;
}

template<typename Scalar, int Dim>
GPU_FUNC bool QuadGLCudaBuffer::setQuad(unsigned int idx, 
                                             const Vector<Scalar, Dim>& v0, 
                                             const Vector<Scalar, Dim>& v1, 
                                             const Vector<Scalar, Dim>& v2, 
                                             const Vector<Scalar, Dim>& v3, 
                                             const Vector<Scalar, 3> & quad_normal)
{
    if (idx >= quadNum())
        return false;

    line_gl_cuda_buffer_.setLine(4 * idx + 0, v0, v1);
    line_gl_cuda_buffer_.setLine(4 * idx + 1, v1, v2);
    line_gl_cuda_buffer_.setLine(4 * idx + 2, v2, v3);
    line_gl_cuda_buffer_.setLine(4 * idx + 3, v3, v0);

    triangle_gl_cuda_buffer_.setTriangle(2 * idx + 0, v0, v1, v2, quad_normal);
    triangle_gl_cuda_buffer_.setTriangle(2 * idx + 1, v0, v2, v3, quad_normal);

    return true;
}

//explicit instantiations
template GPU_FUNC bool QuadGLCudaBuffer::setQuad<float, 2>(unsigned int, const Vector<float, 2> &, const Vector<float, 2> &, const Vector<float, 2> &, const Vector<float, 2> &, bool);
template GPU_FUNC bool QuadGLCudaBuffer::setQuad<float, 3>(unsigned int, const Vector<float, 3> &, const Vector<float, 3> &, const Vector<float, 3> &, const Vector<float, 3> &, bool);
template GPU_FUNC bool QuadGLCudaBuffer::setQuad<double, 2>(unsigned int, const Vector<double, 2> &, const Vector<double, 2> &, const Vector<double, 2> &, const Vector<double, 2> &, bool);
template GPU_FUNC bool QuadGLCudaBuffer::setQuad<double, 3>(unsigned int, const Vector<double, 3> &, const Vector<double, 3> &, const Vector<double, 3> &, const Vector<double, 3> &, bool);

template GPU_FUNC bool QuadGLCudaBuffer::setQuad<float, 2>(unsigned int, const Vector<float, 2> &, const Vector<float, 2> &, const Vector<float, 2> &, const Vector<float, 2> &, const Vector<float, 3> &);
template GPU_FUNC bool QuadGLCudaBuffer::setQuad<float, 3>(unsigned int, const Vector<float, 3> &, const Vector<float, 3> &, const Vector<float, 3> &, const Vector<float, 3> &, const Vector<float, 3> &);
template GPU_FUNC bool QuadGLCudaBuffer::setQuad<double, 2>(unsigned int, const Vector<double, 2> &, const Vector<double, 2> &, const Vector<double, 2> &, const Vector<double, 2> &, const Vector<double, 3> &);
template GPU_FUNC bool QuadGLCudaBuffer::setQuad<double, 3>(unsigned int, const Vector<double, 3> &, const Vector<double, 3> &, const Vector<double, 3> &, const Vector<double, 3> &, const Vector<double, 3> &);

}//end of namespace Physika