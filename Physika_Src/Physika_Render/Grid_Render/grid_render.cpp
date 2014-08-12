/*
 * @file grid_render.cpp 
 * @Basic render of grid.
 * @author Wei Chen
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
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include "Physika_Geometry/Cartesian_Grids/grid.h"
#include "Physika_Render/Grid_Render/grid_render.h"
#include "Physika_Render/Color/color.h"

namespace Physika
{

template <typename Scalar, int Dim>
GridRender<Scalar,Dim>::GridRender()
    :grid_(NULL),
    display_list_id_(0){}

template <typename Scalar, int Dim>
GridRender<Scalar,Dim>::GridRender(const Grid<Scalar,Dim> * grid)
    :grid_(grid),
    display_list_id_(0){}

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
        openGLColor3(Color<Scalar>::White());
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
        openGLColor3(Color<Scalar>::White());
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

template <typename Scalar, int Dim> template <typename ColorType>
void GridRender<Scalar,Dim>::renderNodeWithColor(const std::vector< Vector<unsigned int,Dim> > & node_vec, const Color<ColorType> &color)
{
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT|GL_TEXTURE_BIT|GL_COLOR_BUFFER_BIT|GL_CURRENT_BIT|GL_POINT_BIT);
    glDisable(GL_LIGHTING);                        /// turn light off, otherwise the color may not appear
    openGLColor3(color);
    float point_size;
    glGetFloatv(GL_POINT_SIZE,&point_size);
    glPointSize(static_cast<float>(3*point_size));

    glPushMatrix();
    glBegin(GL_POINTS);
    for(unsigned node_idx = 0; node_idx<node_vec.size(); node_idx ++)
    {
        Vector<Scalar, Dim> pos = this->grid_->node(node_vec[node_idx]);
        openGLVertex(pos);
    }
    glEnd();
    glPopMatrix();
    glPopAttrib();
}

template <typename Scalar, int Dim> template <typename ColorType>
void GridRender<Scalar,Dim>::renderNodeWithColor(const std::vector< Vector<unsigned int,Dim> > & node_vec, const std::vector< Color<ColorType> > &color)
{
    if(node_vec.size()!= color.size())
    {
        std::cerr<<"Warning: the size of node_vec don't equal to color's, the node lacking of cunstom color will be rendered in white color !\n";
    }

    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT|GL_TEXTURE_BIT|GL_COLOR_BUFFER_BIT|GL_CURRENT_BIT|GL_POINT_BIT);
    glDisable(GL_LIGHTING);                        /// turn light off, otherwise the color may not appear
    float point_size;
    glGetFloatv(GL_POINT_SIZE,&point_size);
    glPointSize(static_cast<float>(3*point_size));

    glPushMatrix();
    glBegin(GL_POINTS);
    for(unsigned node_idx = 0; node_idx<node_vec.size(); node_idx ++)
    {
        if(node_idx < color.size())
            openGLColor3(color[node_idx]);
        else
            openGLColor3(Color<ColorType>::White());
        Vector<Scalar, Dim> pos = this->grid_->node(node_vec[node_idx]);
        openGLVertex(pos);
    }
    glEnd();
    glPopMatrix();
    glPopAttrib();
}

template <typename Scalar, int Dim> template <typename ColorType>
void GridRender<Scalar,Dim>::renderCellWithColor(const std::vector< Vector<unsigned int,Dim> > & cell_vec, const Color<ColorType> &color)
{
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT|GL_TEXTURE_BIT|GL_COLOR_BUFFER_BIT|GL_CURRENT_BIT|GL_POINT_BIT|GL_LINE_BIT);
    glDisable(GL_LIGHTING);                        /// turn light off, otherwise the color may not appear

    float line_width;
    glGetFloatv(GL_LINE_WIDTH,&line_width);
    glLineWidth(static_cast<float>(2*line_width));

    openGLColor3(color);
    glPushMatrix();
    for(unsigned int cell_idx = 0; cell_idx<cell_vec.size(); cell_idx++)
    {
        this->renderCell(cell_vec[cell_idx]);
    }
    glPopMatrix();
    glPopAttrib();
}

template <typename Scalar, int Dim> template <typename ColorType>
void GridRender<Scalar,Dim>::renderCellWithColor(const std::vector< Vector<unsigned int,Dim> > & cell_vec, const std::vector< Color<ColorType> > &color)
{
    if(cell_vec.size()!= color.size())
    {
        std::cerr<<"Warning: the size of cell_vec don't equal to color's, the cell lacking of cunstom color will be rendered in white color !\n";
    }

    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT|GL_TEXTURE_BIT|GL_COLOR_BUFFER_BIT|GL_CURRENT_BIT|GL_POINT_BIT|GL_LINE_BIT);
    glDisable(GL_LIGHTING);                        /// turn light off, otherwise the color may not appear

    float line_width;
    glGetFloatv(GL_LINE_WIDTH,&line_width);
    glLineWidth(static_cast<float>(2*line_width));

    glPushMatrix();
    for(unsigned int cell_idx = 0; cell_idx<cell_vec.size(); cell_idx++)
    {
        if(cell_idx < color.size())
            openGLColor3(color[cell_idx]);
        else
            openGLColor3(Color<ColorType>::White());
        this->renderCell(cell_vec[cell_idx]);
    }
    glPopMatrix();
    glPopAttrib();
}

//explicit instantitation
template class GridRender<float,3>;
template class GridRender<double,3>;
template class GridRender<float,2>;
template class GridRender<double,2>;

// renderCellWithColor
template void GridRender<float,3>::renderCellWithColor<signed char>(const std::vector< Vector<unsigned int,3> > &, const Color<signed char> &);
template void GridRender<float,3>::renderCellWithColor<short>(const std::vector< Vector<unsigned int,3> > &, const Color<short> &);
template void GridRender<float,3>::renderCellWithColor<int>(const std::vector< Vector<unsigned int,3> > &, const Color<int> &);
template void GridRender<float,3>::renderCellWithColor<float>(const std::vector< Vector<unsigned int,3> > &, const Color<float> &);
template void GridRender<float,3>::renderCellWithColor<double>(const std::vector< Vector<unsigned int,3> > &, const Color<double> &);
template void GridRender<float,3>::renderCellWithColor<unsigned short>(const std::vector< Vector<unsigned int,3> > &, const Color<unsigned short> &);
template void GridRender<float,3>::renderCellWithColor<unsigned int>(const std::vector< Vector<unsigned int,3> > &, const Color<unsigned int> &);
template void GridRender<float,3>::renderCellWithColor<unsigned char>(const std::vector< Vector<unsigned int,3> > &, const Color<unsigned char> &);

template void GridRender<float,2>::renderCellWithColor<signed char>(const std::vector< Vector<unsigned int,2> > &, const Color<signed char> &);
template void GridRender<float,2>::renderCellWithColor<short>(const std::vector< Vector<unsigned int,2> > &, const Color<short> &);
template void GridRender<float,2>::renderCellWithColor<int>(const std::vector< Vector<unsigned int,2> > &, const Color<int> &);
template void GridRender<float,2>::renderCellWithColor<float>(const std::vector< Vector<unsigned int,2> > &, const Color<float> &);
template void GridRender<float,2>::renderCellWithColor<double>(const std::vector< Vector<unsigned int,2> > &, const Color<double> &);
template void GridRender<float,2>::renderCellWithColor<unsigned short>(const std::vector< Vector<unsigned int,2> > &, const Color<unsigned short> &);
template void GridRender<float,2>::renderCellWithColor<unsigned int>(const std::vector< Vector<unsigned int,2> > &, const Color<unsigned int> &);
template void GridRender<float,2>::renderCellWithColor<unsigned char>(const std::vector< Vector<unsigned int,2> > &, const Color<unsigned char> &);

template void GridRender<float,3>::renderCellWithColor<signed char>(const std::vector< Vector<unsigned int,3> > &, const std::vector< Color<signed char> > &);
template void GridRender<float,3>::renderCellWithColor<short>(const std::vector< Vector<unsigned int,3> > &, const std::vector< Color<short> > &);
template void GridRender<float,3>::renderCellWithColor<int>(const std::vector< Vector<unsigned int,3> > &, const std::vector< Color<int> > &);
template void GridRender<float,3>::renderCellWithColor<float>(const std::vector< Vector<unsigned int,3> > &, const std::vector< Color<float> > &);
template void GridRender<float,3>::renderCellWithColor<double>(const std::vector< Vector<unsigned int,3> > &, const std::vector< Color<double> > &);
template void GridRender<float,3>::renderCellWithColor<unsigned short>(const std::vector< Vector<unsigned int,3> > &, const std::vector< Color<unsigned short> > &);
template void GridRender<float,3>::renderCellWithColor<unsigned int>(const std::vector< Vector<unsigned int,3> > &, const std::vector< Color<unsigned int> > &);
template void GridRender<float,3>::renderCellWithColor<unsigned char>(const std::vector< Vector<unsigned int,3> > &, const std::vector< Color<unsigned char> > &);

template void GridRender<float,2>::renderCellWithColor<signed char>(const std::vector< Vector<unsigned int,2> > &, const std::vector< Color<signed char> > &);
template void GridRender<float,2>::renderCellWithColor<short>(const std::vector< Vector<unsigned int,2> > &, const std::vector< Color<short> > &);
template void GridRender<float,2>::renderCellWithColor<int>(const std::vector< Vector<unsigned int,2> > &, const std::vector< Color<int> > &);
template void GridRender<float,2>::renderCellWithColor<float>(const std::vector< Vector<unsigned int,2> > &, const std::vector< Color<float> > &);
template void GridRender<float,2>::renderCellWithColor<double>(const std::vector< Vector<unsigned int,2> > &, const std::vector< Color<double> > &);
template void GridRender<float,2>::renderCellWithColor<unsigned short>(const std::vector< Vector<unsigned int,2> > &, const std::vector< Color<unsigned short> > &);
template void GridRender<float,2>::renderCellWithColor<unsigned int>(const std::vector< Vector<unsigned int,2> > &, const std::vector< Color<unsigned int> >&);
template void GridRender<float,2>::renderCellWithColor<unsigned char>(const std::vector< Vector<unsigned int,2> > &, const std::vector< Color<unsigned char> > &);

// renderNodeWithColor
template void GridRender<float,3>::renderNodeWithColor<signed char>(const std::vector< Vector<unsigned int,3> > &, const Color<signed char> &);
template void GridRender<float,3>::renderNodeWithColor<short>(const std::vector< Vector<unsigned int,3> > &, const Color<short> &);
template void GridRender<float,3>::renderNodeWithColor<int>(const std::vector< Vector<unsigned int,3> > &, const Color<int> &);
template void GridRender<float,3>::renderNodeWithColor<float>(const std::vector< Vector<unsigned int,3> > &, const Color<float> &);
template void GridRender<float,3>::renderNodeWithColor<double>(const std::vector< Vector<unsigned int,3> > &, const Color<double> &);
template void GridRender<float,3>::renderNodeWithColor<unsigned short>(const std::vector< Vector<unsigned int,3> > &, const Color<unsigned short> &);
template void GridRender<float,3>::renderNodeWithColor<unsigned int>(const std::vector< Vector<unsigned int,3> > &, const Color<unsigned int> &);
template void GridRender<float,3>::renderNodeWithColor<unsigned char>(const std::vector< Vector<unsigned int,3> > &, const Color<unsigned char> &);

template void GridRender<float,2>::renderNodeWithColor<signed char>(const std::vector< Vector<unsigned int,2> > &, const Color<signed char> &);
template void GridRender<float,2>::renderNodeWithColor<short>(const std::vector< Vector<unsigned int,2> > &, const Color<short> &);
template void GridRender<float,2>::renderNodeWithColor<int>(const std::vector< Vector<unsigned int,2> > &, const Color<int> &);
template void GridRender<float,2>::renderNodeWithColor<float>(const std::vector< Vector<unsigned int,2> > &, const Color<float> &);
template void GridRender<float,2>::renderNodeWithColor<double>(const std::vector< Vector<unsigned int,2> > &, const Color<double> &);
template void GridRender<float,2>::renderNodeWithColor<unsigned short>(const std::vector< Vector<unsigned int,2> > &, const Color<unsigned short> &);
template void GridRender<float,2>::renderNodeWithColor<unsigned int>(const std::vector< Vector<unsigned int,2> > &, const Color<unsigned int> &);
template void GridRender<float,2>::renderNodeWithColor<unsigned char>(const std::vector< Vector<unsigned int,2> > &, const Color<unsigned char> &);

template void GridRender<float,3>::renderNodeWithColor<signed char>(const std::vector< Vector<unsigned int,3> > &, const std::vector< Color<signed char> > &);
template void GridRender<float,3>::renderNodeWithColor<short>(const std::vector< Vector<unsigned int,3> > &, const std::vector< Color<short> > &);
template void GridRender<float,3>::renderNodeWithColor<int>(const std::vector< Vector<unsigned int,3> > &, const std::vector< Color<int> > &);
template void GridRender<float,3>::renderNodeWithColor<float>(const std::vector< Vector<unsigned int,3> > &, const std::vector< Color<float> > &);
template void GridRender<float,3>::renderNodeWithColor<double>(const std::vector< Vector<unsigned int,3> > &, const std::vector< Color<double> > &);
template void GridRender<float,3>::renderNodeWithColor<unsigned short>(const std::vector< Vector<unsigned int,3> > &, const std::vector< Color<unsigned short> > &);
template void GridRender<float,3>::renderNodeWithColor<unsigned int>(const std::vector< Vector<unsigned int,3> > &, const std::vector< Color<unsigned int> > &);
template void GridRender<float,3>::renderNodeWithColor<unsigned char>(const std::vector< Vector<unsigned int,3> > &, const std::vector< Color<unsigned char> > &);

template void GridRender<float,2>::renderNodeWithColor<signed char>(const std::vector< Vector<unsigned int,2> > &, const std::vector< Color<signed char> > &);
template void GridRender<float,2>::renderNodeWithColor<short>(const std::vector< Vector<unsigned int,2> > &, const std::vector< Color<short> > &);
template void GridRender<float,2>::renderNodeWithColor<int>(const std::vector< Vector<unsigned int,2> > &, const std::vector< Color<int> > &);
template void GridRender<float,2>::renderNodeWithColor<float>(const std::vector< Vector<unsigned int,2> > &, const std::vector< Color<float> > &);
template void GridRender<float,2>::renderNodeWithColor<double>(const std::vector< Vector<unsigned int,2> > &, const std::vector< Color<double> > &);
template void GridRender<float,2>::renderNodeWithColor<unsigned short>(const std::vector< Vector<unsigned int,2> > &, const std::vector< Color<unsigned short> > &);
template void GridRender<float,2>::renderNodeWithColor<unsigned int>(const std::vector< Vector<unsigned int,2> > &, const std::vector< Color<unsigned int> >&);
template void GridRender<float,2>::renderNodeWithColor<unsigned char>(const std::vector< Vector<unsigned int,2> > &, const std::vector< Color<unsigned char> > &);


} // end of namespace Physika
