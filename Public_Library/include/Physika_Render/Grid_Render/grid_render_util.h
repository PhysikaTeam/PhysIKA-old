/*
 * @file grid_render_util.h 
 * @Basic class GridRenderUtil
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

#include "Physika_Core/Utilities/dimension_trait.h"
#include <memory>

namespace Physika{

class PointRenderUtil;
class QuadRenderUtil;
class CubeRenderUtil;

template <typename Scalar, int Dim> class Grid;

template <typename Scalar, int Dim>
class GridRenderUtil
{
public:
    explicit GridRenderUtil(const Grid<Scalar, Dim> * grid);

    const Grid<Scalar, Dim> * grid() const;
    void setGrid(const Grid<Scalar, Dim> * grid);

    void synchronize();

    unsigned int cellNum() const;
    std::shared_ptr<PointRenderUtil> getInnerPointRenderUtil();
    std::shared_ptr<QuadRenderUtil> getInnerQuadRenderUtil();

private:
    void initPointRenderUtil();

    void initQuadRenderUtil();
    void initQuadRenderUtil(DimensionTrait<2>);
    void initQuadRenderUtil(DimensionTrait<3>);

private:
    const Grid<Scalar, Dim> * grid_ = nullptr;

    std::shared_ptr<PointRenderUtil> point_render_util_;
    std::shared_ptr<QuadRenderUtil> quad_render_util_;

};
    
}//end of namespace Physika