/*
 * @file grid_render-inl.h 
 * @Brief render of grid.
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

#ifndef PHYSIKA_RENDER_GRID_RENDER_GRID_RENDER_INL_H_
#define PHYSIKA_RENDER_GRID_RENDER_GRID_RENDER_INL_H_

#include <GL/gl.h>
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"

namespace Physika{

template <typename Scalar, int Dim>
template <typename ColorType>
void GridRender<Scalar,Dim>::setGridColor(const Color<ColorType> &color)
{
    grid_color_ = color.template convertColor<double>();
}

template <typename Scalar, int Dim>
template <typename ColorType>
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

template <typename Scalar, int Dim> 
template <typename ColorType>
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

template <typename Scalar, int Dim> 
template <typename ColorType>
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

template <typename Scalar, int Dim> 
template <typename ColorType>
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

}  //end of namespace Physika

#endif //PHYSIKA_RENDER_GRID_RENDER_GRID_RENDER_INL_H_
