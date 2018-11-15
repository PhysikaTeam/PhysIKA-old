/*
 * @file grid_render_util.cpp
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

#include "Physika_Render/Point_Render/point_render_util.h"
#include "Physika_Render/Quad_Render/quad_render_util.h"

#include "Physika_Geometry/Cartesian_Grids/grid.h"

#include "grid_render_util.h"

namespace Physika{

template <typename Scalar, int Dim>
GridRenderUtil<Scalar, Dim>::GridRenderUtil(const Grid<Scalar, Dim> * grid)
    :grid_(grid),
    point_render_util_(std::make_shared<PointRenderUtil>()),
    quad_render_util_(std::make_shared<QuadRenderUtil>())
{
    this->initPointRenderUtil();
    this->initQuadRenderUtil();
}

template <typename Scalar, int Dim>
const Grid<Scalar, Dim> * GridRenderUtil<Scalar, Dim>::grid() const
{
    return grid_;
}

template <typename Scalar, int Dim>
void GridRenderUtil<Scalar, Dim>::setGrid(const Grid<Scalar, Dim> * grid)
{
    this->grid_ = grid;
    this->initPointRenderUtil();
    this->initQuadRenderUtil();
}

template <typename Scalar, int Dim>
void GridRenderUtil<Scalar, Dim>::synchronize()
{
    this->initPointRenderUtil();
    this->initQuadRenderUtil();
}

template <typename Scalar, int Dim>
unsigned int GridRenderUtil<Scalar, Dim>::cellNum() const
{
    if (Dim == 2)
        return quad_render_util_->quadNum();
    else
        return quad_render_util_->quadNum() / 6;
}

template <typename Scalar, int Dim>
std::shared_ptr<PointRenderUtil> GridRenderUtil<Scalar, Dim>::getInnerPointRenderUtil()
{
    return point_render_util_;
}

template <typename Scalar, int Dim>
std::shared_ptr<QuadRenderUtil> GridRenderUtil<Scalar, Dim>::getInnerQuadRenderUtil()
{
    return quad_render_util_;
}

    
template <typename Scalar, int Dim>
void GridRenderUtil<Scalar, Dim>::initPointRenderUtil()
{
    std::vector<Vector<Scalar, Dim>> point_vec;
    for(auto node_iter = grid_->nodeBegin(); node_iter != grid_->nodeEnd(); ++node_iter)
    {
        Vector<unsigned int, Dim> node_idx = node_iter.nodeIndex();
        const Vector<Scalar, Dim> & pos = grid_->node(node_idx);
        point_vec.push_back(pos);
    }

    point_render_util_->setPoints(point_vec);
}

template <typename Scalar, int Dim>
void GridRenderUtil<Scalar, Dim>::initQuadRenderUtil()
{
    this->initQuadRenderUtil(DimensionTrait<Dim>());
}

template <typename Scalar, int Dim>
void GridRenderUtil<Scalar, Dim>::initQuadRenderUtil(DimensionTrait<2>)
{
    std::vector<Vector<Scalar, 2>> quad_pos_vec;
    for(auto cell_iter = grid_->cellBegin(); cell_iter != grid_->cellEnd(); ++cell_iter)
    {
        const Vector<unsigned int, Dim> & cell_idx = cell_iter.cellIndex();
        const Vector<Scalar, Dim> & min_pos = grid_->cellMinCornerNode(cell_idx);
        const Vector<Scalar, Dim> & max_pos = grid_->cellMaxCornerNode(cell_idx);

        Vector<Scalar, 2> v0(min_pos[0], min_pos[1]);
        Vector<Scalar, 2> v2(max_pos[0], max_pos[1]);

        Vector<Scalar, 2> v1 = { v2[0], v0[1] };
        Vector<Scalar, 2> v3 = { v0[0], v2[1] };

        quad_pos_vec.push_back(v0);
        quad_pos_vec.push_back(v1);
        quad_pos_vec.push_back(v2);
        quad_pos_vec.push_back(v3);
    }

    quad_render_util_->setQuads(quad_pos_vec);
}

template <typename Scalar, int Dim>
void GridRenderUtil<Scalar, Dim>::initQuadRenderUtil(DimensionTrait<3>)
{
    std::vector<Vector<Scalar, 3>> quad_pos_vec;
    for (auto cell_iter = grid_->cellBegin(); cell_iter != grid_->cellEnd(); ++cell_iter)
    {
        const Vector<unsigned int, Dim> & cell_idx = cell_iter.cellIndex();
        const Vector<Scalar, Dim> & min_pos = grid_->cellMinCornerNode(cell_idx);
        const Vector<Scalar, Dim> & max_pos = grid_->cellMaxCornerNode(cell_idx);

        Vector<Scalar, 3> v0(min_pos[0], min_pos[1], min_pos[2]);
        Vector<Scalar, 3> v6(max_pos[0], max_pos[1], max_pos[2]);

        Vector<Scalar, 3> v1(v6[0], v0[1], v0[2]);
        Vector<Scalar, 3> v2(v6[0], v6[1], v0[2]);
        Vector<Scalar, 3> v3(v0[0], v6[1], v0[2]);
        Vector<Scalar, 3> v4(v0[0], v0[1], v6[2]);
        Vector<Scalar, 3> v5(v6[0], v0[1], v6[2]);
        Vector<Scalar, 3> v7(v0[0], v6[1], v6[2]);

        //face one
        quad_pos_vec.push_back(v0);
        quad_pos_vec.push_back(v3);
        quad_pos_vec.push_back(v2);
        quad_pos_vec.push_back(v1);

        //face two
        quad_pos_vec.push_back(v4);
        quad_pos_vec.push_back(v5);
        quad_pos_vec.push_back(v6);
        quad_pos_vec.push_back(v7);

        //face three
        quad_pos_vec.push_back(v0);
        quad_pos_vec.push_back(v1);
        quad_pos_vec.push_back(v5);
        quad_pos_vec.push_back(v4);

        //face four
        quad_pos_vec.push_back(v3);
        quad_pos_vec.push_back(v7);
        quad_pos_vec.push_back(v6);
        quad_pos_vec.push_back(v2);

        //face five
        quad_pos_vec.push_back(v4);
        quad_pos_vec.push_back(v7);
        quad_pos_vec.push_back(v3);
        quad_pos_vec.push_back(v0);

        //face six
        quad_pos_vec.push_back(v1);
        quad_pos_vec.push_back(v2);
        quad_pos_vec.push_back(v6);
        quad_pos_vec.push_back(v5);
    }

    quad_render_util_->setQuads(quad_pos_vec);
}

//explicit instantiation
template class GridRenderUtil<float, 2>;
template class GridRenderUtil<float, 3>;
template class GridRenderUtil<double, 2>;
template class GridRenderUtil<double, 3>;

}//end of namespace Physika