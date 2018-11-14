/*
 * @file triangle_gl_cuda_buffer.cu
 * @Basic class CudaTriangleGLBuffer
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

#include "triangle_gl_cuda_buffer.h"

namespace Physika{
    
COMM_FUNC TriangleGLCudaBuffer::TriangleGLCudaBuffer(float * pos_dev_ptr, float * normal_dev_ptr, unsigned int triangle_num)
    :pos_dev_ptr_(pos_dev_ptr), normal_dev_ptr_(normal_dev_ptr), triangle_num_(triangle_num)
{
    
}

COMM_FUNC unsigned int TriangleGLCudaBuffer::triangleNum() const
{
    return triangle_num_;
}

COMM_FUNC float * TriangleGLCudaBuffer::getCudaPosPtr()
{
    return pos_dev_ptr_;
}

COMM_FUNC float * TriangleGLCudaBuffer::getCudaNormalPtr()
{
    return normal_dev_ptr_;
}

template<typename Scalar, int Dim>
GPU_FUNC bool TriangleGLCudaBuffer::setTriangle(unsigned int idx, 
                                                     const Vector<Scalar, Dim> & v0, 
                                                     const Vector<Scalar, Dim> & v1, 
                                                     const Vector<Scalar, Dim> & v2, 
                                                     bool auto_compute_normal)
{
    if (idx >= triangle_num_)
        return false;

    setTrianglePos(idx, v0, v1, v2);

    if(auto_compute_normal == true)
    {
        //Note: unified processing for Dim = 2/3
        Vector<Scalar, 3> fir_edge, sec_edge;
        for(int i = 0; i < Dim; ++i)
        {
            fir_edge[i] = v1[i] - v0[i];
            sec_edge[i] = v2[i] - v0[i];
        }

        Vector<Scalar, 3> normal = fir_edge.cross(sec_edge);
        normal.normalize();

        setTriangleNormal(idx, normal, normal, normal);
    }

    return true;
}

template<typename Scalar, int Dim>
GPU_FUNC bool TriangleGLCudaBuffer::setTriangle(unsigned int idx, 
                                                     const Vector<Scalar, Dim> & v0, 
                                                     const Vector<Scalar, Dim> & v1, 
                                                     const Vector<Scalar, Dim> & v2, 
                                                     const Vector<Scalar, 3> & triangle_normal)
{
    return setTriangle(idx, v0, v1, v2, triangle_normal, triangle_normal, triangle_normal);
}

template<typename Scalar, int Dim>
GPU_FUNC bool TriangleGLCudaBuffer::setTriangle(unsigned int idx, 
                                                     const Vector<Scalar, Dim> & v0, 
                                                     const Vector<Scalar, Dim> & v1, 
                                                     const Vector<Scalar, Dim> & v2, 
                                                     const Vector<Scalar, 3> & v0_normal, 
                                                     const Vector<Scalar, 3> & v1_normal, 
                                                     const Vector<Scalar, 3> & v2_normal)
{
    if (idx >= triangle_num_)
        return false;

    setTrianglePos(idx, v0, v1, v2);
    setTriangleNormal(idx, v0_normal, v1_normal, v2_normal);

    return true;
}

template<typename Scalar, int Dim>
GPU_FUNC void TriangleGLCudaBuffer::setTrianglePos(unsigned int idx, 
                                                        const Vector<Scalar, Dim>& v0, 
                                                        const Vector<Scalar, Dim>& v1, 
                                                        const Vector<Scalar, Dim>& v2)
{
    pos_dev_ptr_[9 * idx + 0] = v0[0];
    pos_dev_ptr_[9 * idx + 1] = v0[1];
    pos_dev_ptr_[9 * idx + 2] = v0[2];

    pos_dev_ptr_[9 * idx + 3] = v1[0];
    pos_dev_ptr_[9 * idx + 4] = v1[1];
    pos_dev_ptr_[9 * idx + 5] = v1[2];

    pos_dev_ptr_[9 * idx + 6] = v2[0];
    pos_dev_ptr_[9 * idx + 7] = v2[1];
    pos_dev_ptr_[9 * idx + 8] = v2[2];
}

template<typename Scalar>
GPU_FUNC void TriangleGLCudaBuffer::setTriangleNormal(unsigned int idx, 
                                                           const Vector<Scalar, 3>& v0_normal,
                                                           const Vector<Scalar, 3>& v1_normal, 
                                                           const Vector<Scalar, 3>& v2_normal)
{
    normal_dev_ptr_[9 * idx + 0] = v0_normal[0];
    normal_dev_ptr_[9 * idx + 1] = v0_normal[1];
    normal_dev_ptr_[9 * idx + 2] = v0_normal[2];

    normal_dev_ptr_[9 * idx + 3] = v1_normal[0];
    normal_dev_ptr_[9 * idx + 4] = v1_normal[1];
    normal_dev_ptr_[9 * idx + 5] = v1_normal[2];

    normal_dev_ptr_[9 * idx + 6] = v2_normal[0];
    normal_dev_ptr_[9 * idx + 7] = v2_normal[1];
    normal_dev_ptr_[9 * idx + 8] = v2_normal[2];
}

//explicit instantiations
template GPU_FUNC bool TriangleGLCudaBuffer::setTriangle<float, 2>(unsigned int, const Vector<float, 2> &, const Vector<float, 2> &, const Vector<float, 2> &, bool);
template GPU_FUNC bool TriangleGLCudaBuffer::setTriangle<float, 3>(unsigned int, const Vector<float, 3> &, const Vector<float, 3> &, const Vector<float, 3> &, bool);
template GPU_FUNC bool TriangleGLCudaBuffer::setTriangle<double, 2>(unsigned int, const Vector<double, 2> &, const Vector<double, 2> &, const Vector<double, 2> &, bool);
template GPU_FUNC bool TriangleGLCudaBuffer::setTriangle<double, 3>(unsigned int, const Vector<double, 3> &, const Vector<double, 3> &, const Vector<double, 3> &, bool);


template GPU_FUNC bool TriangleGLCudaBuffer::setTriangle<float, 2>(unsigned int, const Vector<float, 2> &, const Vector<float, 2> &, const Vector<float, 2> &, const Vector<float, 3> &);
template GPU_FUNC bool TriangleGLCudaBuffer::setTriangle<float, 3>(unsigned int, const Vector<float, 3> &, const Vector<float, 3> &, const Vector<float, 3> &, const Vector<float, 3> &);
template GPU_FUNC bool TriangleGLCudaBuffer::setTriangle<double, 2>(unsigned int, const Vector<double, 2> &, const Vector<double, 2> &, const Vector<double, 2> &, const Vector<double, 3> &);
template GPU_FUNC bool TriangleGLCudaBuffer::setTriangle<double, 3>(unsigned int, const Vector<double, 3> &, const Vector<double, 3> &, const Vector<double, 3> &, const Vector<double, 3> &);

template GPU_FUNC bool TriangleGLCudaBuffer::setTriangle<float, 2>(unsigned int, const Vector<float, 2> &, const Vector<float, 2> &, const Vector<float, 2> &, const Vector<float, 3> &, const Vector<float, 3> &, const Vector<float, 3> &);
template GPU_FUNC bool TriangleGLCudaBuffer::setTriangle<float, 3>(unsigned int, const Vector<float, 3> &, const Vector<float, 3> &, const Vector<float, 3> &, const Vector<float, 3> &, const Vector<float, 3> &, const Vector<float, 3> &);
template GPU_FUNC bool TriangleGLCudaBuffer::setTriangle<double, 2>(unsigned int, const Vector<double, 2> &, const Vector<double, 2> &, const Vector<double, 2> &, const Vector<double, 3> &, const Vector<double, 3> &, const Vector<double, 3> &);
template GPU_FUNC bool TriangleGLCudaBuffer::setTriangle<double, 3>(unsigned int, const Vector<double, 3> &, const Vector<double, 3> &, const Vector<double, 3> &, const Vector<double, 3> &, const Vector<double, 3> &, const Vector<double, 3> &);

template GPU_FUNC void TriangleGLCudaBuffer::setTrianglePos<float, 2>(unsigned int, const Vector<float, 2> &, const Vector<float, 2> &, const Vector<float, 2> &);
template GPU_FUNC void TriangleGLCudaBuffer::setTrianglePos<float, 3>(unsigned int, const Vector<float, 3> &, const Vector<float, 3> &, const Vector<float, 3> &);
template GPU_FUNC void TriangleGLCudaBuffer::setTrianglePos<double, 2>(unsigned int, const Vector<double, 2> &, const Vector<double, 2> &, const Vector<double, 2> &);
template GPU_FUNC void TriangleGLCudaBuffer::setTrianglePos<double, 3>(unsigned int, const Vector<double, 3> &, const Vector<double, 3> &, const Vector<double, 3> &);

template GPU_FUNC void TriangleGLCudaBuffer::setTriangleNormal<float>(unsigned int, const Vector<float, 3> &, const Vector<float, 3> &, const Vector<float, 3> &);
template GPU_FUNC void TriangleGLCudaBuffer::setTriangleNormal<double>(unsigned int, const Vector<double, 3> &, const Vector<double, 3> &, const Vector<double, 3> &);

}//end of namespace Physika