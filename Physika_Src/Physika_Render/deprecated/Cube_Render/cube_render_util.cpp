/*
 * @file cube_render_util.h 
 * @Basic class CubeRenderUtil
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

#include "Physika_Render/Quad_Render/quad_gl_cuda_buffer.h"
#include "cube_gl_cuda_buffer.h"

#include "cube_render_util.h"

namespace Physika{

CubeRenderUtil::CubeRenderUtil()
    :quad_render_util_(std::make_shared<QuadRenderUtil>())
{
    
}

template <typename Scalar>
void CubeRenderUtil::setCubes(const std::vector<Vector<Scalar, 3>> & pos_vec, bool auto_compute_normal /* = true */)
{
    if (pos_vec.size() % 8 != 0)
        throw PhysikaException("error: pos_vec.size() % 8 != 0!");

    std::vector<Vector<Scalar, 3>> quad_pos_vec;
    std::vector<Vector<Scalar, 3>> line_pos_vec;

    for (unsigned int i = 0; i < pos_vec.size(); i += 8)
    {
        const Vector<Scalar, 3> & v0 = pos_vec[i];
        const Vector<Scalar, 3> & v1 = pos_vec[i + 1];
        const Vector<Scalar, 3> & v2 = pos_vec[i + 2];
        const Vector<Scalar, 3> & v3 = pos_vec[i + 3];
        const Vector<Scalar, 3> & v4 = pos_vec[i + 4];
        const Vector<Scalar, 3> & v5 = pos_vec[i + 5];
        const Vector<Scalar, 3> & v6 = pos_vec[i + 6];
        const Vector<Scalar, 3> & v7 = pos_vec[i + 7];

        //face one
        quad_pos_vec.push_back(v0);
        quad_pos_vec.push_back(v3);
        quad_pos_vec.push_back(v2);
        quad_pos_vec.push_back(v1);

        //face two
        quad_pos_vec.push_back(v4);
        quad_pos_vec.push_back(v5);
        quad_pos_vec.push_back(v6);
        quad_pos_vec.push_back(v7);

        //face three
        quad_pos_vec.push_back(v0);
        quad_pos_vec.push_back(v1);
        quad_pos_vec.push_back(v5);
        quad_pos_vec.push_back(v4);

        //face four
        quad_pos_vec.push_back(v3);
        quad_pos_vec.push_back(v7);
        quad_pos_vec.push_back(v6);
        quad_pos_vec.push_back(v2);

        //face five
        quad_pos_vec.push_back(v4);
        quad_pos_vec.push_back(v7);
        quad_pos_vec.push_back(v3);
        quad_pos_vec.push_back(v0);

        //face six
        quad_pos_vec.push_back(v1);
        quad_pos_vec.push_back(v2);
        quad_pos_vec.push_back(v6);
        quad_pos_vec.push_back(v5);
    }

    quad_render_util_->setQuads(quad_pos_vec, auto_compute_normal);
}

template <typename Scalar>
void CubeRenderUtil::setCubes(const std::vector<Vector<Scalar, 3>> & pos_vec, std::vector<unsigned int> & indices, bool auto_compute_normal /* = true */)
{
    if (pos_vec.size() < 8 || indices.size() < 8 || indices.size() % 8 != 0)
        throw PhysikaException("error: pos_vec.size() < 8 || indices.size() < 8|| indices.size() % 8 != 0!");

    std::vector<unsigned int> quad_indices_vec;

    for (unsigned int i = 0; i < indices.size(); i += 8)
    {
        unsigned int v0 = indices[i];
        unsigned int v1 = indices[i + 1];
        unsigned int v2 = indices[i + 2];
        unsigned int v3 = indices[i + 3];
        unsigned int v4 = indices[i + 4];
        unsigned int v5 = indices[i + 5];
        unsigned int v6 = indices[i + 6];
        unsigned int v7 = indices[i + 7];

        ///face one
        quad_indices_vec.push_back(v0);
        quad_indices_vec.push_back(v3);
        quad_indices_vec.push_back(v2);
        quad_indices_vec.push_back(v1);

        //face two
        quad_indices_vec.push_back(v4);
        quad_indices_vec.push_back(v5);
        quad_indices_vec.push_back(v6);
        quad_indices_vec.push_back(v7);

        //face three
        quad_indices_vec.push_back(v0);
        quad_indices_vec.push_back(v1);
        quad_indices_vec.push_back(v5);
        quad_indices_vec.push_back(v4);

        //face four
        quad_indices_vec.push_back(v3);
        quad_indices_vec.push_back(v7);
        quad_indices_vec.push_back(v6);
        quad_indices_vec.push_back(v2);

        //face five
        quad_indices_vec.push_back(v4);
        quad_indices_vec.push_back(v7);
        quad_indices_vec.push_back(v3);
        quad_indices_vec.push_back(v0);

        //face six
        quad_indices_vec.push_back(v1);
        quad_indices_vec.push_back(v2);
        quad_indices_vec.push_back(v6);
        quad_indices_vec.push_back(v5);
    }

    quad_render_util_->setQuads(pos_vec, quad_indices_vec, auto_compute_normal);
}

template <typename Scalar>
void CubeRenderUtil::setNormals(const std::vector<Vector<Scalar, 3>> & normals)
{
    quad_render_util_->setNormals(normals);
}

CubeGLCudaBuffer CubeRenderUtil::mapCubeGLCudaBuffer(unsigned int cube_num)
{
    QuadGLCudaBuffer quad_gl_cuda_buffer = quad_render_util_->mapQuadGLCudaBuffer(6 * cube_num);
    return CubeGLCudaBuffer(quad_gl_cuda_buffer);
}

void CubeRenderUtil::unmapCubeGLCudaBuffer()
{
    quad_render_util_->unmapQuadGLCudaBuffer();
}

unsigned int CubeRenderUtil::cubeNum() const
{
    return quad_render_util_->quadNum() / 6;
}

std::shared_ptr<QuadRenderUtil> CubeRenderUtil::getInnerQuadRenderUtil()
{
    return quad_render_util_;
}

void CubeRenderUtil::drawCubeLine()
{
    quad_render_util_->drawQuadLine();
}

void CubeRenderUtil::drawCube()
{
    quad_render_util_->drawQuad();
}

void CubeRenderUtil::bindCubeLineVAO()
{
    quad_render_util_->bindQuadLineVAO();
}

void CubeRenderUtil::unbindCubeLineVAO()
{
    quad_render_util_->unbindQuadLineVAO();
}

void CubeRenderUtil::bindCubeVAO()
{
    quad_render_util_->bindQuadVAO();
}
   
void CubeRenderUtil::unbindCubeVAO()
{
    quad_render_util_->unbindQuadVAO();
}

//explicit instantiation
template void CubeRenderUtil::setCubes<float>(const std::vector<Vector<float, 3>> & pos_vec, bool auto_compute_normal /* = true */);
template void CubeRenderUtil::setCubes<double>(const std::vector<Vector<double, 3>> & pos_vec, bool auto_compute_normal /* = true */);
template void CubeRenderUtil::setCubes<float>(const std::vector<Vector<float, 3>> & pos_vec, std::vector<unsigned int> & indices, bool auto_compute_normal /* = true */);
template void CubeRenderUtil::setCubes<double>(const std::vector<Vector<double, 3>> & pos_vec, std::vector<unsigned int> & indices, bool auto_compute_normal /* = true */);
template void CubeRenderUtil::setNormals<float>(const std::vector<Vector<float, 3>> & normals);
template void CubeRenderUtil::setNormals<double>(const std::vector<Vector<double, 3>> & normals);

}//end of namespace Physika