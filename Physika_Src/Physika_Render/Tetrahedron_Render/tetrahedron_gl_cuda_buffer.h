/*
 * @file tetrahedron_gl_cuda_buffer.h 
 * @Basic class TetrahedronGLCudaBuffer
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
 *  TetrahedronGLCudaBuffer defines the cuda memory format to set the tet data mapped from opengl VBO. 
 */

class TetrahedronGLCudaBuffer
{
public:
    CPU_GPU_FUNC_DECL TetrahedronGLCudaBuffer(TriangleGLCudaBuffer cuda_triangle_gl_buffer);

    CPU_GPU_FUNC_DECL unsigned int tetrahedrondNum() const;

    template<typename Scalar>
    GPU_FUNC_DECL bool setTetrahedron(unsigned int idx, 
                               const Vector<Scalar, 3> & v0, 
                               const Vector<Scalar, 3> & v1, 
                               const Vector<Scalar, 3> & v2, 
                               const Vector<Scalar, 3> & v3, 
                               bool auto_compute_normal = true);

    template<typename Scalar>
    GPU_FUNC_DECL bool setTetrahedron(unsigned int idx,
                               const Vector<Scalar, 3> & v0,
                               const Vector<Scalar, 3> & v1,
                               const Vector<Scalar, 3> & v2,
                               const Vector<Scalar, 3> & v3,
                               const Vector<Scalar, 3> & face_normal_0, //face v0-v2-v1
                               const Vector<Scalar, 3> & face_normal_1, //face v0-v1-v3
                               const Vector<Scalar, 3> & face_normal_2, //face v1-v2-v3
                               const Vector<Scalar, 3> & face_normal_3);//face v0-v3-v2

private:
    TriangleGLCudaBuffer triangle_gl_cuda_buffer_;
    
};

}//end of namespace Physika