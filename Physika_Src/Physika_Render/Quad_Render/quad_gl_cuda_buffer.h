/*
 * @file quad_gl_cuda_buffer.h 
 * @Basic class QuadGLCudaBuffer
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
#include "Physika_Render/Triangle_Render/triangle_gl_cuda_buffer.h"

namespace Physika{

template <typename Scalar, int Dim> class Vector;

/*
 *  QuadGLCudaBuffer defines the cuda memory format to set the quad data mapped from opengl VBO. 
 */

class QuadGLCudaBuffer
{
public:
    CPU_GPU_FUNC_DECL QuadGLCudaBuffer(LineGLCudaBuffer cuda_line_gl_buffer, TriangleGLCudaBuffer cuda_triangle_gl_buffer);

    CPU_GPU_FUNC_DECL unsigned int quadNum() const;

    template<typename Scalar, int Dim>
    GPU_FUNC_DECL bool setQuad(unsigned int idx, 
                               const Vector<Scalar, Dim> & v0, 
                               const Vector<Scalar, Dim> & v1, 
                               const Vector<Scalar, Dim> & v2, 
                               const Vector<Scalar, Dim> & v3, 
                               bool auto_compute_normal = true);

    template<typename Scalar, int Dim>
    GPU_FUNC_DECL bool setQuad(unsigned int idx,
                               const Vector<Scalar, Dim> & v0,
                               const Vector<Scalar, Dim> & v1,
                               const Vector<Scalar, Dim> & v2,
                               const Vector<Scalar, Dim> & v3,
                               const Vector<Scalar, 3> & quad_normal);

private:
    LineGLCudaBuffer line_gl_cuda_buffer_;
    TriangleGLCudaBuffer triangle_gl_cuda_buffer_;
    
};

}//end of namespace Physika