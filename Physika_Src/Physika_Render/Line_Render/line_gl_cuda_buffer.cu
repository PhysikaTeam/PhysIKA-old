/*
 * @file line_gl_cuda_buffer.cu
 * @Basic class CudaLineGLBuffer
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

#include "line_gl_cuda_buffer.h"

namespace Physika{

CPU_GPU_FUNC_DECL LineGLCudaBuffer::LineGLCudaBuffer(float * line_data_dev_ptr, unsigned int line_num)
    :line_data_dev_ptr_(line_data_dev_ptr), line_num_(line_num)
{
    
}

CPU_GPU_FUNC_DECL unsigned int LineGLCudaBuffer::lineNum() const
{
    return line_num_;
}

CPU_GPU_FUNC_DECL float * LineGLCudaBuffer::getCudaPosPtr()
{
    return line_data_dev_ptr_;
}

template <typename Scalar, int Dim>
GPU_FUNC_DECL bool LineGLCudaBuffer::setLine(unsigned int idx, const Vector<Scalar, Dim> & fir_point, const Vector<Scalar, Dim> & sec_point)
{
    if (Dim == 2)
        return setLine(idx, fir_point[0], fir_point[1], static_cast<Scalar>(0), sec_point[0], sec_point[1], static_cast<Scalar>(0));

    return setLine(idx, fir_point[0], fir_point[1], fir_point[2], sec_point[0], sec_point[1], sec_point[2]);
}

template <typename Scalar>
GPU_FUNC_DECL bool LineGLCudaBuffer::setLine(unsigned int idx, Scalar x1, Scalar y1, Scalar x2, Scalar y2)
{
    return setLine(idx, x1, y1, static_cast<Scalar>(0), x2, y2, static_cast<Scalar>(0));
}

template <typename Scalar>
GPU_FUNC_DECL bool LineGLCudaBuffer::setLine(unsigned int idx, Scalar x1, Scalar y1, Scalar z1, Scalar x2, Scalar y2, Scalar z2)
{
    if (idx >= line_num_)
        return false;

    line_data_dev_ptr_[6 * idx    ] = x1;
    line_data_dev_ptr_[6 * idx + 1] = y1;
    line_data_dev_ptr_[6 * idx + 2] = z1;

    line_data_dev_ptr_[6 * idx + 3] = x2;
    line_data_dev_ptr_[6 * idx + 4] = y2;
    line_data_dev_ptr_[6 * idx + 5] = z2;

    return true;
}

//explicit instantiations
template GPU_FUNC_DECL bool LineGLCudaBuffer::setLine<float, 2>(unsigned int idx, const Vector<float, 2> &, const Vector<float, 2> &);
template GPU_FUNC_DECL bool LineGLCudaBuffer::setLine<float, 3>(unsigned int idx, const Vector<float, 3> &, const Vector<float, 3> &);
template GPU_FUNC_DECL bool LineGLCudaBuffer::setLine<double, 2>(unsigned int idx, const Vector<double, 2> &, const Vector<double, 2> &);
template GPU_FUNC_DECL bool LineGLCudaBuffer::setLine<double, 3>(unsigned int idx, const Vector<double, 3> &, const Vector<double, 3> &);

template GPU_FUNC_DECL bool LineGLCudaBuffer::setLine<float>(unsigned int idx, float, float, float, float);
template GPU_FUNC_DECL bool LineGLCudaBuffer::setLine<double>(unsigned int idx, double, double, double, double);

template GPU_FUNC_DECL bool LineGLCudaBuffer::setLine<float>(unsigned int idx, float, float, float, float, float, float);
template GPU_FUNC_DECL bool LineGLCudaBuffer::setLine<double>(unsigned int idx, double, double, double, double, double, double);

}//end of namespace Physika