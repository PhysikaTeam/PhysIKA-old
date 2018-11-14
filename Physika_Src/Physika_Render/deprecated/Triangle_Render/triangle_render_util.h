/*
 * @file triangle_render_util.h 
 * @Basic class TriangleRenderUtil
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

#include <vector>
#include <memory>
#include <glm/glm.hpp>

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Utilities/physika_exception.h"


namespace Physika {

class VBOCudaMapper;
class TriangleGLCudaBuffer;

class TriangleRenderUtil
{
public:
    TriangleRenderUtil();
    ~TriangleRenderUtil();

    //disable copy
    TriangleRenderUtil(const TriangleRenderUtil &) = delete;
    TriangleRenderUtil & operator = (const TriangleRenderUtil &) = delete;

    //enable move
    TriangleRenderUtil(TriangleRenderUtil &&) noexcept;
    TriangleRenderUtil & operator = (TriangleRenderUtil &&) noexcept;

    template <typename Scalar, int Dim>
    void setTriangles(const std::vector<Vector<Scalar, Dim>> & pos_vec, bool auto_compute_normal = true);

    template <typename Scalar, int Dim>
    void setTriangles(const std::vector<Vector<Scalar, Dim>> & pos_vec, std::vector<unsigned int> & indices, bool auto_compute_normal = true);

    //Note: normals.size() = 3 * triangle_num
    template <typename Scalar>
    void setNormalsPerVertex(const std::vector<Vector<Scalar, 3>> & normals);

    //Note: normals.size() = triangle_num
    template <typename Scalar>
    void setNormalsPerTriangle(const std::vector<Vector<Scalar, 3>> & normals);

    template <typename Scalar>
    void setTexCoords(const std::vector<Vector<Scalar, 2>> & tex_coords);

    //Note: triangle_num = 0 means that you want maintain the pre-set triangle_num.    
    TriangleGLCudaBuffer mapTriangleGLCudaBuffer(unsigned int triangle_num = 0);
    void unmapTriangleGLCudaBuffer();

    unsigned int triangleNum() const;
    void draw();

    void bindTriangleVAO();
    void unbindTriangleVAO();

private:
    void initVAOAndVBOs();
    void destoryVAOAndVBOs();

    void updateTriangleVBOData(const std::vector<glm::vec3> & glm_pos_vec);
    void updateNormalVBOData(const std::vector<glm::vec3> & glm_normal_vec);
    void updateTexCoordVBOData(const std::vector<glm::vec2> & glm_tex_coord_vec);

    void autoComputeNormal(const std::vector<glm::vec3> & glm_pos_vec);

private:
    unsigned int triangle_num_ = 0;
    unsigned int triangle_VBO_ = 0;
    unsigned int normal_VBO_ = 0;
    unsigned int tex_coord_VBO_ = 0;
    unsigned int triangle_VAO_ = 0;

    std::shared_ptr<VBOCudaMapper> pos_vbo_mapper_;
    std::shared_ptr<VBOCudaMapper> normal_vbo_mapper_;
};

}//end of namespace Physika