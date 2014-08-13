/*
 * @file grid_render.cpp 
 * @Basic render of grid.
 * @author Wei Chen, Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <cstddef>
#include <iostream>
#include <GL/gl.h>
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include "Physika_Geometry/Cartesian_Grids/grid.h"
#include "Physika_Render/Grid_Render/grid_render.h"

namespace Physika
{

template <typename Scalar, int Dim>
GridRender<Scalar,Dim>::GridRender()
    :grid_(NULL),
    display_list_id_(0)
{
    grid_color_ = Color<double>::White();
}

template <typename Scalar, int Dim>
GridRender<Scalar,Dim>::GridRender(const Grid<Scalar,Dim> * grid)
    :grid_(grid),
    display_list_id_(0)
{
    grid_color_ = Color<double>::White();
}

template <typename Scalar, int Dim>
GridRender<Scalar,Dim>::~GridRender()
{
    this->display_list_id_ = 0;
    this->grid_ = NULL;
}

template <typename Scalar, int Dim>
const Grid<Scalar,Dim> * GridRender<Scalar, Dim>::grid() const
{
    return this->grid_;
}

template <typename Scalar, int Dim>
void GridRender<Scalar,Dim>::setGrid(const Grid<Scalar, Dim> * grid)
{
    this->grid_ = grid;
}

template <typename Scalar, int Dim>
void GridRender<Scalar,Dim>::synchronize()
{
    glDeleteLists(this->display_list_id_, 1);
    this->display_list_id_ = 0;
}

template <typename Scalar, int Dim>
void GridRender<Scalar,Dim>::printInfo()const
{
    std::cout<<"grid address: "<<this->grid_<<std::endl;
    std::cout<<"display list id: "<< this->display_list_id_<<std::endl;
}

template <typename Scalar, int Dim>
void GridRender<Scalar,Dim>::renderCell(const Vector<unsigned int, Dim> & cell_idx)
{
    if(Dim == 2)
    {
        Vector<Scalar, Dim> min_corner_node = this->grid_->cellMinCornerNode(cell_idx);
        Vector<Scalar, Dim> max_corner_node = this->grid_->cellMaxCornerNode(cell_idx);
        openGLColor3(grid_color_);
        glBegin(GL_LINE_LOOP);
        openGLVertex(min_corner_node);
        openGLVertex(Vector<Scalar,2>(max_corner_node[0], min_corner_node[1]));
        openGLVertex(max_corner_node);
        openGLVertex(Vector<Scalar,2>(min_corner_node[0],max_corner_node[1]));
        glEnd();
    }
    else if(Dim == 3)
    {
        Vector<Scalar, Dim> min_corner_node = this->grid_->cellMinCornerNode(cell_idx);
        Vector<Scalar, Dim> max_corner_node = this->grid_->cellMaxCornerNode(cell_idx);
        Vector<Scalar, 3> node_1(max_corner_node[0], min_corner_node[1], min_corner_node[2]);
        Vector<Scalar, 3> node_2(max_corner_node[0], max_corner_node[1], min_corner_node[2]);
        Vector<Scalar, 3> node_3(min_corner_node[0], max_corner_node[1], min_corner_node[2]);
        Vector<Scalar, 3> node_4(min_corner_node[0], min_corner_node[1], max_corner_node[2]);
        Vector<Scalar, 3> node_5(max_corner_node[0], min_corner_node[1], max_corner_node[2]);
        Vector<Scalar, 3> node_7(min_corner_node[0], max_corner_node[1], max_corner_node[2]);
        openGLColor3(grid_color_);
        // face one
        glBegin(GL_LINE_LOOP);
        openGLVertex(min_corner_node);
        openGLVertex(node_1);
        openGLVertex(node_5);
        openGLVertex(node_4);
        glEnd();

        // face two
        glBegin(GL_LINE_LOOP);
        openGLVertex(max_corner_node);
        openGLVertex(node_2);
        openGLVertex(node_3);
        openGLVertex(node_7);
        glEnd();

        // face three
        glBegin(GL_LINE_LOOP);
        openGLVertex(node_1);
        openGLVertex(node_2);
        openGLVertex(max_corner_node);
        openGLVertex(node_5);
        glEnd();

        // face four
        glBegin(GL_LINE_LOOP);
        openGLVertex(node_4);
        openGLVertex(node_7);
        openGLVertex(node_3);
        openGLVertex(min_corner_node);
        glEnd();

        // face five
        glBegin(GL_LINE_LOOP);
        openGLVertex(node_4);
        openGLVertex(node_5);
        openGLVertex(max_corner_node);
        openGLVertex(node_7);
        glEnd();

        // face six
        glBegin(GL_LINE_LOOP);
        openGLVertex(min_corner_node);
        openGLVertex(node_3);
        openGLVertex(node_2);
        openGLVertex(node_1);
        glEnd();
    }
}
template <typename Scalar, int Dim>
void GridRender<Scalar,Dim>::render()
{
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT);
    glDisable(GL_LIGHTING);
    glPushMatrix();
    if(! glIsList(this->display_list_id_))
    {
        this->display_list_id_ = glGenLists(1);
        glNewList(this->display_list_id_, GL_COMPILE_AND_EXECUTE);
        GridCellIterator<Scalar,Dim> cell_iter = this->grid_->cellBegin();
        while(cell_iter != this->grid_->cellEnd())
        {
            Vector<unsigned int, Dim> cell_idx = cell_iter.cellIndex();
            this->renderCell(cell_idx);
            cell_iter++;
        }
        glEndList();
    }
    else
    {
        glCallList(this->display_list_id_);
    }
    glPopMatrix();
    glPopAttrib();
}

//explicit instantitationtemplate class Grid<unsigned char,2>;
template class GridRender<unsigned short,2>;
template class GridRender<unsigned int,2>;
template class GridRender<unsigned long,2>;
template class GridRender<unsigned long long,2>;
template class GridRender<signed char,2>;
template class GridRender<short,2>;
template class GridRender<int,2>;
template class GridRender<long,2>;
template class GridRender<long long,2>;
template class GridRender<float,2>;
template class GridRender<double,2>;
template class GridRender<long double,2>;
template class GridRender<unsigned char,3>;
template class GridRender<unsigned short,3>;
template class GridRender<unsigned int,3>;
template class GridRender<unsigned long,3>;
template class GridRender<unsigned long long,3>;
template class GridRender<signed char,3>;
template class GridRender<short,3>;
template class GridRender<int,3>;
template class GridRender<long,3>;
template class GridRender<long long,3>;
template class GridRender<float,3>;
template class GridRender<double,3>;
template class GridRender<long double,3>;

} // end of namespace Physika
