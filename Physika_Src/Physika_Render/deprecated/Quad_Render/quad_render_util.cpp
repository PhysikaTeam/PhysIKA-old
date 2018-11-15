/*
 * @file quad_render_util.h 
 * @Basic class QuadRenderUtil
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

#include "quad_render_util.h"
#include "quad_gl_cuda_buffer.h"

namespace Physika{

QuadRenderUtil::QuadRenderUtil()
    :line_render_util_(std::make_shared<LineRenderUtil>()),
    triangle_render_util_(std::make_shared<TriangleRenderUtil>())
{
    
}

template <typename Scalar, int Dim>
void QuadRenderUtil::setQuads(const std::vector<Vector<Scalar, Dim>> & pos_vec, bool auto_compute_normal /* = true */)
{
    if (pos_vec.size() % 4 != 0)
        throw PhysikaException("error: pos_vec.size() % 4 != 0!");

    std::vector<Vector<Scalar, Dim>> triangle_pos_vec;
    std::vector<Vector<Scalar, Dim>> line_pos_vec;

    for (unsigned int i = 0; i < pos_vec.size(); i += 4)
    {
        const Vector<Scalar, Dim> & v0 = pos_vec[i];
        const Vector<Scalar, Dim> & v1 = pos_vec[i + 1];
        const Vector<Scalar, Dim> & v2 = pos_vec[i + 2];
        const Vector<Scalar, Dim> & v3 = pos_vec[i + 3];

        //face one
        triangle_pos_vec.push_back(v0);
        triangle_pos_vec.push_back(v1);
        triangle_pos_vec.push_back(v2);

        //face two
        triangle_pos_vec.push_back(v0);
        triangle_pos_vec.push_back(v2);
        triangle_pos_vec.push_back(v3);

        //line one
        line_pos_vec.push_back(v0);
        line_pos_vec.push_back(v1);

        //line two
        line_pos_vec.push_back(v1);
        line_pos_vec.push_back(v2);

        //line three
        line_pos_vec.push_back(v2);
        line_pos_vec.push_back(v3);

        //line four
        line_pos_vec.push_back(v3);
        line_pos_vec.push_back(v0);
    }

    triangle_render_util_->setTriangles(triangle_pos_vec, auto_compute_normal);
    line_render_util_->setLinePairs(line_pos_vec);
}

template <typename Scalar, int Dim>
void QuadRenderUtil::setQuads(const std::vector<Vector<Scalar, Dim>> & pos_vec, std::vector<unsigned int> & indices, bool auto_compute_normal /* = true */)
{
    if (pos_vec.size() < 4 || indices.size() < 4 || indices.size() % 4 != 0)
        throw PhysikaException("error: pos_vec.size() < 4 || indices.size() < 4|| indices.size() % 4 != 0!");

    std::vector<unsigned int> triangle_indices_vec;
    std::vector<unsigned int> line_indices_vec;

    for (unsigned int i = 0; i < indices.size(); i += 4)
    {
        unsigned int v0 = indices[i];
        unsigned int v1 = indices[i + 1];
        unsigned int v2 = indices[i + 2];
        unsigned int v3 = indices[i + 3];

        //face one
        triangle_indices_vec.push_back(v0);
        triangle_indices_vec.push_back(v1);
        triangle_indices_vec.push_back(v2);

        //face two
        triangle_indices_vec.push_back(v0);
        triangle_indices_vec.push_back(v2);
        triangle_indices_vec.push_back(v3);

        //line one
        line_indices_vec.push_back(v0);
        line_indices_vec.push_back(v1);

        //line two
        line_indices_vec.push_back(v1);
        line_indices_vec.push_back(v2);

        //line three
        line_indices_vec.push_back(v2);
        line_indices_vec.push_back(v3);

        //line four
        line_indices_vec.push_back(v3);
        line_indices_vec.push_back(v0);
    }

    triangle_render_util_->setTriangles(pos_vec, triangle_indices_vec, auto_compute_normal);
    line_render_util_->setLines(pos_vec, line_indices_vec);
}

template <typename Scalar>
void QuadRenderUtil::setNormals(const std::vector<Vector<Scalar, 3>> & normals)
{
    std::vector<Vector<Scalar, 3>> triangle_normals;
    for (unsigned int i = 0; i < normals.size(); ++i)
        triangle_normals.insert(triangle_normals.end(), 2, normals[i]);

    triangle_render_util_->setNormalsPerTriangle(triangle_normals);
}

QuadGLCudaBuffer QuadRenderUtil::mapQuadGLCudaBuffer(unsigned int quad_num)
{
    LineGLCudaBuffer line_gl_cuda_buffer = line_render_util_->mapLineGLCudaBuffer(4 * quad_num);
    TriangleGLCudaBuffer triangle_gl_cuda_buffer = triangle_render_util_->mapTriangleGLCudaBuffer(2 * quad_num);

    return QuadGLCudaBuffer(line_gl_cuda_buffer, triangle_gl_cuda_buffer);
}

void QuadRenderUtil::unmapQuadGLCudaBuffer()
{
    line_render_util_->unmapLineGLCudaBuffer();
    triangle_render_util_->unmapTriangleGLCudaBuffer();
}

unsigned int QuadRenderUtil::quadNum() const
{
    return triangle_render_util_->triangleNum() / 2;
}

std::shared_ptr<LineRenderUtil> QuadRenderUtil::getInnerLineRenderUtil()
{
    return line_render_util_;
}

std::shared_ptr<TriangleRenderUtil> QuadRenderUtil::getInnerTriangleRenderUtil()
{
    return triangle_render_util_;
}

void QuadRenderUtil::drawQuadLine()
{
    line_render_util_->draw();
}

void QuadRenderUtil::drawQuad()
{
    triangle_render_util_->draw();
}

void QuadRenderUtil::bindQuadLineVAO()
{
    line_render_util_->bindLineVAO();
}

void QuadRenderUtil::unbindQuadLineVAO()
{
    line_render_util_->unbindLineVAO();
}

void QuadRenderUtil::bindQuadVAO()
{
    triangle_render_util_->bindTriangleVAO();
}
   
void QuadRenderUtil::unbindQuadVAO()
{
    triangle_render_util_->unbindTriangleVAO();
}

//explicit instantiation
template void QuadRenderUtil::setQuads<float, 2>(const std::vector<Vector<float, 2>> & pos_vec, bool auto_compute_normal /* = true */);
template void QuadRenderUtil::setQuads<float, 3>(const std::vector<Vector<float, 3>> & pos_vec, bool auto_compute_normal /* = true */);
template void QuadRenderUtil::setQuads<double, 2>(const std::vector<Vector<double, 2>> & pos_vec, bool auto_compute_normal /* = true */);
template void QuadRenderUtil::setQuads<double, 3>(const std::vector<Vector<double, 3>> & pos_vec, bool auto_compute_normal /* = true */);

template void QuadRenderUtil::setQuads<float, 2>(const std::vector<Vector<float, 2>> & pos_vec, std::vector<unsigned int> & indices, bool auto_compute_normal /* = true */);
template void QuadRenderUtil::setQuads<float, 3>(const std::vector<Vector<float, 3>> & pos_vec, std::vector<unsigned int> & indices, bool auto_compute_normal /* = true */);
template void QuadRenderUtil::setQuads<double, 2>(const std::vector<Vector<double, 2>> & pos_vec, std::vector<unsigned int> & indices, bool auto_compute_normal /* = true */);
template void QuadRenderUtil::setQuads<double, 3>(const std::vector<Vector<double, 3>> & pos_vec, std::vector<unsigned int> & indices, bool auto_compute_normal /* = true */);

template void QuadRenderUtil::setNormals<float>(const std::vector<Vector<float, 3>> & normals);
template void QuadRenderUtil::setNormals<double>(const std::vector<Vector<double, 3>> & normals);

}//end of namespace Physika