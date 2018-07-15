/*
 * @file triangle_gl_cuda_buffer.h 
 * @Basic class TriangleGLCudaBuffer
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
 *  TriangleGLCudaBuffer defines the cuda memory format to set the triangle data mapped from opengl VBO. 
 */

class TriangleGLCudaBuffer
{
public:
    COMM_FUNC TriangleGLCudaBuffer(float * pos_dev_ptr, float * normal_dev_ptr, unsigned int triangle_num);

    COMM_FUNC unsigned int triangleNum() const;
    COMM_FUNC float * getCudaPosPtr();      //return the raw position pointer, each triangle uses 9 float nums representing (x1, y1, z1), (x2, y2, z2), (x3, y3, z3).
    COMM_FUNC float * getCudaNormalPtr();   //return the raw normal pointer, each triangle uses 9 float nums representing (nx1, ny1, nz1), (nx2, ny2, nz2), (nx3, ny3, nz3);


    template<typename Scalar, int Dim>
    GPU_FUNC bool setTriangle(unsigned int idx, 
                                   const Vector<Scalar, Dim> & v0, 
                                   const Vector<Scalar, Dim> & v1, 
                                   const Vector<Scalar, Dim> & v2, 
                                   bool auto_compute_normal = true);

    template<typename Scalar, int Dim>
    GPU_FUNC bool setTriangle(unsigned int idx,
                                   const Vector<Scalar, Dim> & v0,
                                   const Vector<Scalar, Dim> & v1,
                                   const Vector<Scalar, Dim> & v2,
                                   const Vector<Scalar, 3> & triangle_normal);

    template<typename Scalar, int Dim>
    GPU_FUNC bool setTriangle(unsigned int idx,
                                   const Vector<Scalar, Dim> & v0,
                                   const Vector<Scalar, Dim> & v1,
                                   const Vector<Scalar, Dim> & v2,
                                   const Vector<Scalar, 3> & v0_normal,
                                   const Vector<Scalar, 3> & v1_normal,
                                   const Vector<Scalar, 3> & v2_normal);

private:
    template<typename Scalar, int Dim>
    GPU_FUNC void setTrianglePos(unsigned int idx, 
                                      const Vector<Scalar, Dim> & v0, 
                                      const Vector<Scalar, Dim> & v1,
                                      const Vector<Scalar, Dim> & v2);

    template<typename Scalar>
    GPU_FUNC void setTriangleNormal(unsigned int idx, 
                                         const Vector<Scalar, 3> & v0_normal,
                                         const Vector<Scalar, 3> & v1_normal,
                                         const Vector<Scalar, 3> & v2_normal);

private:
    float * pos_dev_ptr_;
    float * normal_dev_ptr_;
    unsigned int triangle_num_;
    
};

}//end of namespace Physika