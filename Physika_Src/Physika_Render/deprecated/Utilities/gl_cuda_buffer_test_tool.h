/*
 * @file gl_cuda_buffer_tool.h 
 * @Basic Test tool for ***GLCudaBuffer
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

/*
 * Note: The functions defined in this file are only for test and should not be used by user.
 */

#pragma once

#include <vector>
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

class PointGLCudaBuffer;
class LineGLCudaBuffer;
class TriangleGLCudaBuffer;

class TetrahedronGLCudaBuffer;
class QuadGLCudaBuffer;
class CubeGLCudaBuffer;

void setPointGLCudaBuffer(const std::vector<Vector3f> & point_data, PointGLCudaBuffer & point_gl_cuda_buffer);
void setLineGLCudaBuffer(const std::vector<Vector3f> & line_data, LineGLCudaBuffer & line_gl_cuda_buffer);
void setTriangleGLCudaBuffer(const std::vector<Vector3f> & triangle_data, TriangleGLCudaBuffer & triangle_gl_cuda_buffer);


void setTetrahedronGLCudaBuffer(const std::vector<Vector3f> & tet_data, TetrahedronGLCudaBuffer & tet_gl_cuda_buffer);
void setQuadGLCudaBuffer(const std::vector<Vector3f> & quad_data, QuadGLCudaBuffer & quad_gl_cuda_buffer);
void setCubeGLCudaBuffer(const std::vector<Vector3f> & cube_data, CubeGLCudaBuffer & cube_gl_cuda_buffer);

}//end of namespace Physika