/*
 * @file point_gl_cuda_buffer.cu
 * @Basic class CudaPointGLBuffer
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

#include "point_gl_cuda_buffer.h"


namespace Physika{

COMM_FUNC PointGLCudaBuffer::PointGLCudaBuffer(float * point_data_dev_ptr, unsigned int point_num)
    :point_data_dev_ptr_(point_data_dev_ptr), point_num_(point_num)
{    
}

COMM_FUNC float * PointGLCudaBuffer::getCudaPosPtr()
{
    return point_data_dev_ptr_;
}

COMM_FUNC unsigned int PointGLCudaBuffer::pointNum() const
{
    return point_num_;
}
    
template<typename Scalar, int Dim>
GPU_FUNC bool PointGLCudaBuffer::setPoint(unsigned int idx, const Vector<Scalar, Dim> & pos)
{
    if (Dim == 2)
        return setPoint(idx, pos[0], pos[1], static_cast<Scalar>(0));

    return setPoint(idx, pos[0], pos[1], pos[2]);
}

template<typename Scalar>
GPU_FUNC bool PointGLCudaBuffer::setPoint(unsigned int idx, Scalar x, Scalar y, Scalar z = 0)
{
    if (idx >= point_num_)
        return false;

    point_data_dev_ptr_[3 * idx] = x;
    point_data_dev_ptr_[3 * idx + 1] = y;
    point_data_dev_ptr_[3 * idx + 2] = z;

    return true;
}

//explicit instantiations
template GPU_FUNC bool PointGLCudaBuffer::setPoint<float, 2>(unsigned int, const Vector<float, 2> &);
template GPU_FUNC bool PointGLCudaBuffer::setPoint<float, 3>(unsigned int, const Vector<float, 3> &);
template GPU_FUNC bool PointGLCudaBuffer::setPoint<double, 2>(unsigned int, const Vector<double, 2> &);
template GPU_FUNC bool PointGLCudaBuffer::setPoint<double, 3>(unsigned int, const Vector<double, 3> &);

template GPU_FUNC bool PointGLCudaBuffer::setPoint<float>(unsigned int, float, float, float);
template GPU_FUNC bool PointGLCudaBuffer::setPoint<double>(unsigned int, double, double, double);

}//end of namespace Physika