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

#pragma once

#include <vector>
#include <memory>

#include "Physika_Core/Vectors/vector_3d.h"

#include "Physika_Render/Line_Render/line_render_util.h"
#include "Physika_Render/Triangle_Render/triangle_render_util.h"

namespace Physika {

class QuadRenderUtil
{
public:
    QuadRenderUtil();
    ~QuadRenderUtil() = default;

    //disable copy
    QuadRenderUtil(const QuadRenderUtil &) = delete;
    QuadRenderUtil & operator = (const QuadRenderUtil &) = delete;

    template <typename Scalar, int Dim>
    void setQuads(const std::vector<Vector<Scalar, Dim>> & pos_vec, bool auto_compute_normal = true);

    template <typename Scalar, int Dim>
    void setQuads(const std::vector<Vector<Scalar, Dim>> & pos_vec, std::vector<unsigned int> & indices, bool auto_compute_normal = true);

    //Note: normal num = quad num
    template <typename Scalar>
    void setNormals(const std::vector<Vector<Scalar, 3>> & normals);

    unsigned int quadNum() const;
    std::shared_ptr<LineRenderUtil> getInnerLineRenderUtil();
    std::shared_ptr<TriangleRenderUtil> getInnerTriangleRenderUtil();
    
    void drawQuadLine();
    void drawQuad();

    void bindQuadLineVAO();
    void unbindQuadLineVAO();

    void bindQuadVAO();
    void unbindQuadVAO();


private:
    std::shared_ptr<LineRenderUtil> line_render_util_;
    std::shared_ptr<TriangleRenderUtil>  triangle_render_util_;
};

}//end of namespace Physika