/*
 * @file grid_cell_render_task.cpp
 * @Basic cell render task of grid
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

#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"

#include "grid_render_util.h"
#include "grid_cell_render_task.h"

namespace Physika{

template<typename Scalar, int Dim>
GridCellRenderTask<Scalar, Dim>::GridCellRenderTask(std::shared_ptr<GridRenderUtil<Scalar, Dim>> render_util)
    :QuadWireframeRenderTask(render_util->getInnerQuadRenderUtil()),
    render_util_(std::move(render_util))
{

}    

template<typename Scalar, int Dim>
void GridCellRenderTask<Scalar, Dim>::setCellColors(const std::vector<Color4f>& colors)
{
    if (colors.size() != render_util_->cellNum())
        throw PhysikaException("error: color size not match grid num!");

    if(Dim == 3)
    {
        std::vector<Color4f> quad_colors;
        for (unsigned int i = 0; i < colors.size(); ++i)
            quad_colors.insert(quad_colors.end(), 6, colors[i]);

        this->setQuadColors(quad_colors);
        return;
    }

    this->setQuadColors(colors);
}

template<typename Scalar, int Dim>
void GridCellRenderTask<Scalar, Dim>::setElementColors(const std::vector<Color4f>& colors)
{
    this->setCellColors(colors);
}

//explicit instantiations
template class GridCellRenderTask<float, 2>;
template class GridCellRenderTask<float, 3>;
template class GridCellRenderTask<double, 2>;
template class GridCellRenderTask<double, 3>;

}//end of namespace Physika