/*
 * @file point_gl_cuda_buffer.h 
 * @Basic class PointGLCudaBuffer
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

#include "Physika_Core/Utilities/cuda_utilities.h"

namespace Physika{

template <typename Scalar, int Dim> class Vector;

/*
 *  PointGLCudaBuffer defines the cuda memory format to set the point pos data mapped from opengl VBO. 
 */

class PointGLCudaBuffer
{
public:
    CPU_GPU_FUNC_DECL PointGLCudaBuffer(float * point_data_dev_ptr, unsigned int point_num);

    CPU_GPU_FUNC_DECL unsigned int pointNum() const;
    CPU_GPU_FUNC_DECL float * getCudaPosPtr(); //return the raw pointer, each point position uses 3 float nums representing (x, y, z).

    template<typename Scalar, int Dim>
    GPU_FUNC_DECL bool setPoint(unsigned int idx, const Vector<Scalar, Dim> & pos);

    template<typename Scalar>
    GPU_FUNC_DECL bool setPoint(unsigned int idx, Scalar x, Scalar y, Scalar z = 0);

private:
    float * point_data_dev_ptr_;
    unsigned int point_num_;
};

}//end of namespace Physika