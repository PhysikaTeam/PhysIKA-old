/*
 * @file line_render_util.h 
 * @Basic class LineRenderUtil
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


namespace Physika{

class VBOCudaMapper;
class LineGLCudaBuffer;

class LineRenderUtil
{
public:
    LineRenderUtil();
    ~LineRenderUtil();

    //disable copy
    LineRenderUtil(const LineRenderUtil &) = delete;
    LineRenderUtil & operator = (const LineRenderUtil &) = delete;

    template <typename Scalar, int Dim>
    void setLinePairs(const std::vector<Vector<Scalar, Dim>> & line_pairs);
    
    template <typename Scalar, int Dim>
    void setLineStrips(const std::vector<Vector<Scalar, Dim>> & line_strips);

    template <typename Scalar, int Dim>
    void setLines(const std::vector<Vector<Scalar, Dim>> & pos_vec, std::vector<unsigned int> & indices);

    //Note: line_num = 0 means that you want maintain the pre-set line_num.  
    LineGLCudaBuffer mapLineGLCudaBuffer(unsigned int line_num = 0);
    void unmapLineGLCudaBuffer();

    unsigned int lineNum() const;
    void draw();

    void bindLineVAO();
    void unbindLineVAO();

private:
    void initVAOAndVBO();
    void destoryVAOAndVBO();
    void updateLineVBOData(const std::vector<glm::vec3> & glm_pos_vec);

private:
    unsigned int line_num_ = 0;
    unsigned int line_VBO_ = 0;
    unsigned int line_VAO_ = 0;

    std::shared_ptr<VBOCudaMapper> cuda_vbo_mapper_;
};
    
}//end of namespace Physika