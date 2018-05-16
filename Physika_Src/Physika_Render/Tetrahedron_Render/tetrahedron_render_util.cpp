/*
 * @file tetrahedron_render_util.cpp 
 * @Basic class TetrahedronRenderUtil
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

#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"
#include "Physika_Render/Triangle_Render/triangle_gl_cuda_buffer.h"
#include "tetrahedron_gl_cuda_buffer.h"
#include "tetrahedron_render_util.h"

namespace Physika{

TetrahedronRenderUtil::TetrahedronRenderUtil()
    :triangle_render_util_(std::make_shared<TriangleRenderUtil>())
{
    
}

TetrahedronRenderUtil::~TetrahedronRenderUtil()
{
   
}

template <typename Scalar>
void TetrahedronRenderUtil::setTetrahedrons(const std::vector<Vector<Scalar, 3>> & pos_vec, bool auto_compute_normal)
{
    if (pos_vec.size() % 4 != 0)
        throw PhysikaException("error: pos_vec.size() % 4 != 0!");

    std::vector<Vector<Scalar, 3>> triangle_pos_vec;
    for (unsigned int i = 0; i < pos_vec.size(); i += 4)
    {
        const Vector<Scalar, 3> & v0 = pos_vec[i];
        const Vector<Scalar, 3> & v1 = pos_vec[i + 1];
        const Vector<Scalar, 3> & v2 = pos_vec[i + 2];
        const Vector<Scalar, 3> & v3 = pos_vec[i + 3];

        //face one
        triangle_pos_vec.push_back(v0);
        triangle_pos_vec.push_back(v2);
        triangle_pos_vec.push_back(v1);

        //face two
        triangle_pos_vec.push_back(v0);
        triangle_pos_vec.push_back(v1);
        triangle_pos_vec.push_back(v3);

        //face three
        triangle_pos_vec.push_back(v1);
        triangle_pos_vec.push_back(v2);
        triangle_pos_vec.push_back(v3);

        //face four
        triangle_pos_vec.push_back(v0);
        triangle_pos_vec.push_back(v3);
        triangle_pos_vec.push_back(v2);
    }

    triangle_render_util_->setTriangles(triangle_pos_vec, auto_compute_normal);
}

template <typename Scalar>
void TetrahedronRenderUtil::setTetrahedrons(const std::vector<Vector<Scalar, 3>> & pos_vec, std::vector<unsigned int> & indices, bool auto_compute_normal)
{
    if (pos_vec.size() < 4 || indices.size() < 4 || indices.size() % 4 != 0)
        throw PhysikaException("error: pos_vec.size() < 4 || indices.size() < 4|| indices.size() % 4 != 0!");

    std::vector<unsigned int> triangle_indices_vec;
    for (unsigned int i = 0; i < indices.size(); i += 4)
    {
        unsigned int v0 = indices[i];
        unsigned int v1 = indices[i + 1];
        unsigned int v2 = indices[i + 2];
        unsigned int v3 = indices[i + 3];

        //face one
        triangle_indices_vec.push_back(v0);
        triangle_indices_vec.push_back(v2);
        triangle_indices_vec.push_back(v1);

        //face two
        triangle_indices_vec.push_back(v0);
        triangle_indices_vec.push_back(v1);
        triangle_indices_vec.push_back(v3);

        //face three
        triangle_indices_vec.push_back(v1);
        triangle_indices_vec.push_back(v2);
        triangle_indices_vec.push_back(v3);

        //face four
        triangle_indices_vec.push_back(v0);
        triangle_indices_vec.push_back(v3);
        triangle_indices_vec.push_back(v2);
    }

    triangle_render_util_->setTriangles(pos_vec, triangle_indices_vec, auto_compute_normal);
}

template <typename Scalar>
void TetrahedronRenderUtil::setNormals(const std::vector<Vector<Scalar, 3>> & normals)
{
    triangle_render_util_->setNormalsPerTriangle(normals);
}

TetrahedronGLCudaBuffer TetrahedronRenderUtil::mapTetrahedronGLCudaBuffer(unsigned int tet_num)
{
    TriangleGLCudaBuffer triangle_gl_cuda_buffer = triangle_render_util_->mapTriangleGLCudaBuffer(4 * tet_num);
    return TetrahedronGLCudaBuffer(triangle_gl_cuda_buffer);
}

void TetrahedronRenderUtil::unmapTetrahedronGLCudaBuffer()
{
    triangle_render_util_->unmapTriangleGLCudaBuffer();
}

unsigned int TetrahedronRenderUtil::tetrahedronNum() const
{
    return triangle_render_util_->triangleNum() / 4;
}

void TetrahedronRenderUtil::draw()
{
    triangle_render_util_->draw();
}

std::shared_ptr<TriangleRenderUtil> TetrahedronRenderUtil::getInnerTriangleRenderUtil()
{
    return triangle_render_util_;
}

void TetrahedronRenderUtil::bindTetrahedronVAO()
{
    triangle_render_util_->bindTriangleVAO();
}

void TetrahedronRenderUtil::unbindTetrahedronVAO()
{
    triangle_render_util_->unbindTriangleVAO();
}

//explicit instantiation
template void TetrahedronRenderUtil::setTetrahedrons<float>(const std::vector<Vector<float, 3>> & pos_vec, bool auto_compute_normal /* = true */);
template void TetrahedronRenderUtil::setTetrahedrons<double>(const std::vector<Vector<double, 3>> & pos_vec, bool auto_compute_normal /* = true */);
template void TetrahedronRenderUtil::setTetrahedrons<float>(const std::vector<Vector<float, 3>> & pos_vec, std::vector<unsigned int> & indices, bool auto_compute_normal /* = true */);
template void TetrahedronRenderUtil::setTetrahedrons<double>(const std::vector<Vector<double, 3>> & pos_vec, std::vector<unsigned int> & indices, bool auto_compute_normal /* = true */);
template void TetrahedronRenderUtil::setNormals<float>(const std::vector<Vector<float, 3>> & normals);
template void TetrahedronRenderUtil::setNormals<double>(const std::vector<Vector<double, 3>> & normals);

}//end of namespace Physika