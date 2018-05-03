/*
 * @file grid_node_render_task.h 
 * @Basic grid node render task.
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

#include "Physika_Render/Point_Render/point_render_task.h"

namespace Physika{

template <typename Scalar, int Dim>
class GridRenderUtil;

template <typename Scalar, int Dim>
class GridNodeRenderTask: public PointRenderTask
{
public:
    explicit GridNodeRenderTask(std::shared_ptr<GridRenderUtil<Scalar, Dim>> render_util);
    ~GridNodeRenderTask() = default;

    //disable copy
    GridNodeRenderTask(const GridNodeRenderTask &) = delete;
    GridNodeRenderTask & operator = (const GridNodeRenderTask &) = delete;

    void setNodeColors(const std::vector<Color4f> & node_colors);

private:
    std::shared_ptr<GridRenderUtil<Scalar, Dim>> render_util_;
};
    
}//end of namespace Physika