/*
 * @file point_render_util.h 
 * @Basic class PointRenderUtil
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
#include <glm/glm.hpp>

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Render/Color/color.h"

namespace Physika{

class PointRenderUtil
{
public:
    PointRenderUtil();
    ~PointRenderUtil();

    //disable copy
    PointRenderUtil(const PointRenderUtil &) = delete;
    PointRenderUtil & operator = (const PointRenderUtil &) = delete;

    template <typename Scalar, int Dim>
    void setPoints(const std::vector<Vector<Scalar, Dim>> & pos_vec);

    unsigned int pointNum() const;
    void draw();

    void bindPointVAO();
    void unbindPointVAO();

private:
    void initVAOAndVBO();
    void destoryVAOAndVBO();
    void updatePositionVBOData(const std::vector<glm::vec3> & glm_pos_vec);

private:
    unsigned int point_num_ = 0;
    unsigned int pos_VBO_ = 0;
    unsigned int point_VAO_ = 0;
};
    
}//end of namespace Physika