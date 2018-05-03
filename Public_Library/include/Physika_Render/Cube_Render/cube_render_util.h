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

#pragma once

#include <vector>
#include <memory>
#include <glm/glm.hpp>

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Utilities/physika_exception.h"

#include "Physika_Render/Quad_Render/quad_render_util.h"


namespace Physika {

class CubeRenderUtil
{
public:
    CubeRenderUtil();
    ~CubeRenderUtil() = default;

    //disable copy
    CubeRenderUtil(const CubeRenderUtil &) = delete;
    CubeRenderUtil & operator = (const CubeRenderUtil &) = delete;

    template <typename Scalar>
    void setCubes(const std::vector<Vector<Scalar, 3>> & pos_vec, bool auto_compute_normal = true);

    template <typename Scalar>
    void setCubes(const std::vector<Vector<Scalar, 3>> & pos_vec, std::vector<unsigned int> & indices, bool auto_compute_normal = true);

    //Note: normal_num = 6 * cube num, per normal for one face
    template <typename Scalar>
    void setNormals(const std::vector<Vector<Scalar, 3>> & normals);

    unsigned int cubeNum() const;
    std::shared_ptr<QuadRenderUtil> getInnerQuadRenderUtil();

    void drawCubeLine();
    void drawCube();

    void bindCubeLineVAO();
    void unbindCubeLineVAO();

    void bindCubeVAO();
    void unbindCubeVAO();

private:
    std::shared_ptr<QuadRenderUtil>  quad_render_util_;
};

}//end of namespace Physika