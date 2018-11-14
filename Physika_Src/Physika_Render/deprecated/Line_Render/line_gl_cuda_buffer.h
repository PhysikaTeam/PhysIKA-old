/*
 * @file line_gl_cuda_buffer.h 
 * @Basic class LineGLCudaBuffer
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
 *  LineGLCudaBuffer defines the cuda memory format to set the lines pos data mapped from opengl VBO. 
 */

class LineGLCudaBuffer
{
public:
    COMM_FUNC LineGLCudaBuffer(float * line_data_dev_ptr, unsigned int line_num);

    COMM_FUNC unsigned int lineNum() const;
    COMM_FUNC float * getCudaPosPtr(); //return the raw pointer, each line uses 6 float nums representing (x1, y1, z1), (x2, y2, z2).

    template<typename Scalar, int Dim>
    GPU_FUNC bool setLine(unsigned int idx, const Vector<Scalar, Dim> & fir_point, const Vector<Scalar, Dim> & sec_point);

    template<typename Scalar>
    GPU_FUNC bool setLine(unsigned int idx, Scalar x1, Scalar y1, Scalar x2, Scalar y2);

    template<typename Scalar>
    GPU_FUNC bool setLine(unsigned int dix, Scalar x1, Scalar y1, Scalar z1, Scalar x2, Scalar y2, Scalar z2);

private:
    float * line_data_dev_ptr_;
    unsigned int line_num_;
    
};

}//end of namespace Physika