/*
 * @file cube_gl_cuda_buffer.h 
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

#pragma once

#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Render/Line_Render/line_gl_cuda_buffer.h"
#include "Physika_Render/Quad_Render/quad_gl_cuda_buffer.h"

namespace Physika{

template <typename Scalar, int Dim> class Vector;

/*
 *  CubeGLCudaBuffer defines the cuda memory format to set the cube data mapped from opengl VBO. 
 */

class CubeGLCudaBuffer
{
public:
    COMM_FUNC CubeGLCudaBuffer(QuadGLCudaBuffer cuda_triangle_gl_buffer);

    COMM_FUNC unsigned int cubeNum() const;

    template<typename Scalar>
    GPU_FUNC bool setCube(unsigned int idx, 
                               const Vector<Scalar, 3> & v0, 
                               const Vector<Scalar, 3> & v1, 
                               const Vector<Scalar, 3> & v2, 
                               const Vector<Scalar, 3> & v3, 
                               const Vector<Scalar, 3> & v4,
                               const Vector<Scalar, 3> & v5,
                               const Vector<Scalar, 3> & v6,
                               const Vector<Scalar, 3> & v7,
                               bool auto_compute_normal = true);

    template<typename Scalar>
    GPU_FUNC bool setCube(unsigned int idx,
                               const Vector<Scalar, 3> & v0,
                               const Vector<Scalar, 3> & v1,
                               const Vector<Scalar, 3> & v2,
                               const Vector<Scalar, 3> & v3,
                               const Vector<Scalar, 3> & v4,
                               const Vector<Scalar, 3> & v5,
                               const Vector<Scalar, 3> & v6,
                               const Vector<Scalar, 3> & v7,
                               const Vector<Scalar, 3> & face_normal_0, //face v0-v3-v2-v1
                               const Vector<Scalar, 3> & face_normal_1, //face v4-v5-v6-v7
                               const Vector<Scalar, 3> & face_normal_2, //face v0-v1-v5-v4
                               const Vector<Scalar, 3> & face_normal_3, //face v3-v7-v6-v2
                               const Vector<Scalar, 3> & face_normal_4, //face v4-v7-v3-v0
                               const Vector<Scalar, 3> & face_normal_5);//face v1-v2-v6-v5

private:
    QuadGLCudaBuffer quad_gl_cuda_buffer_;
    
};

}//end of namespace Physika