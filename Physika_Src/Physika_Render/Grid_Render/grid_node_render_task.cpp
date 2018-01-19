/*
 * @file grid_node_render_task.cpp
 * @Basic node render task of cubic mesh
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

#include "grid_render_util.h"
#include "grid_node_render_task.h"

namespace Physika{

template <typename Scalar, int Dim>
GridNodeRenderTask<Scalar, Dim>::GridNodeRenderTask(std::shared_ptr<GridRenderUtil<Scalar, Dim>> render_util)
    :PointRenderTask(render_util->getInnerPointRenderUtil()),
    render_util_(render_util)
{
    
}

template <typename Scalar, int Dim>
void GridNodeRenderTask<Scalar, Dim>::setNodeColors(const std::vector<Color4f> & node_colors)
{
    this->setPointColors(node_colors);
}

//explicit instantiation 
template class GridNodeRenderTask<float, 2>;
template class GridNodeRenderTask<float, 3>;
template class GridNodeRenderTask<double, 2>;
template class GridNodeRenderTask<double, 3>;
    
}//end of namespace Physika