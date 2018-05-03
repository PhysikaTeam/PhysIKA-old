/*
 * @file cube_wireframe_render_task.h 
 * @Basic wireframe render task of cube
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

#include <memory>

#include "Physika_Render/Quad_Render/quad_wireframe_render_task.h"


namespace Physika{

template <typename Scalar, int Dim>
class GridRenderUtil;

template <typename Scalar, int Dim>
class GridCellRenderTask: public QuadWireframeRenderTask
{
public:
    explicit GridCellRenderTask(std::shared_ptr<GridRenderUtil<Scalar, Dim>> render_util);
    ~GridCellRenderTask() = default;

    //disable copy
    GridCellRenderTask(const GridCellRenderTask &) = delete;
    GridCellRenderTask & operator = (const GridCellRenderTask &) = delete;

    void setCellColors(const std::vector<Color4f> & colors);
    void setElementColors(const std::vector<Color4f> & colors) override;

private:
    std::shared_ptr<GridRenderUtil<Scalar, Dim>> render_util_;
};

}//end of namespace Physika    