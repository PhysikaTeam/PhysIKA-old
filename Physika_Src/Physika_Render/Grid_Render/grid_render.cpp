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
	:grid_(NULL),display_list_id_(0){}

template <typename Scalar, int Dim>
GridRender<Scalar,Dim>::GridRender(Grid<Scalar,Dim> * grid)
	:grid_(grid),display_list_id_(0){}

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
void GridRender<Scalar,Dim>::setGrid(Grid<Scalar, Dim> * grid)
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
}

template <typename Scalar, int Dim>
void GridRender<Scalar,Dim>::render()
{
}

template <typename Scalar, int Dim> template <typename ColorType>
void GridRender<Scalar,Dim>::renderNodeWithColor(const std::vector< Vector<Scalar,Dim> > & node_vec, const Color<ColorType> &color)
{
}

template <typename Scalar, int Dim> template <typename ColorType>
void GridRender<Scalar,Dim>::renderNodeWithColor(const std::vector< Vector<Scalar,Dim> > & node_vec, const std::vector< Color<ColorType> > &color)
{
}

template <typename Scalar, int Dim> template <typename ColorType>
void GridRender<Scalar,Dim>::renderCellWithColor(const std::vector< Vector<Scalar,Dim> > & node_vec, const Color<ColorType> &color)
{
}

template <typename Scalar, int Dim> template <typename ColorType>
void GridRender<Scalar,Dim>::renderCellWithColor(const std::vector< Vector<Scalar,Dim> > & node_vec, const std::vector< Color<ColorType> > &color)
{
}

//explicit instantitation
template class GridRender<float,3>;
template class GridRender<double,3>;
template class GridRender<float,2>;
template class GridRender<double,2>;

// renderCellWithColor
template void GridRender<float,3>::renderCellWithColor<signed char>(const std::vector< Vector<float,3> > &, const Color<signed char> &);
template void GridRender<float,3>::renderCellWithColor<short>(const std::vector< Vector<float,3> > &, const Color<short> &);
template void GridRender<float,3>::renderCellWithColor<int>(const std::vector< Vector<float,3> > &, const Color<int> &);
template void GridRender<float,3>::renderCellWithColor<float>(const std::vector< Vector<float,3> > &, const Color<float> &);
template void GridRender<float,3>::renderCellWithColor<double>(const std::vector< Vector<float,3> > &, const Color<double> &);
template void GridRender<float,3>::renderCellWithColor<unsigned short>(const std::vector< Vector<float,3> > &, const Color<unsigned short> &);
template void GridRender<float,3>::renderCellWithColor<unsigned int>(const std::vector< Vector<float,3> > &, const Color<unsigned int> &);
template void GridRender<float,3>::renderCellWithColor<unsigned char>(const std::vector< Vector<float,3> > &, const Color<unsigned char> &);

template void GridRender<float,2>::renderCellWithColor<signed char>(const std::vector< Vector<float,2> > &, const Color<signed char> &);
template void GridRender<float,2>::renderCellWithColor<short>(const std::vector< Vector<float,2> > &, const Color<short> &);
template void GridRender<float,2>::renderCellWithColor<int>(const std::vector< Vector<float,2> > &, const Color<int> &);
template void GridRender<float,2>::renderCellWithColor<float>(const std::vector< Vector<float,2> > &, const Color<float> &);
template void GridRender<float,2>::renderCellWithColor<double>(const std::vector< Vector<float,2> > &, const Color<double> &);
template void GridRender<float,2>::renderCellWithColor<unsigned short>(const std::vector< Vector<float,2> > &, const Color<unsigned short> &);
template void GridRender<float,2>::renderCellWithColor<unsigned int>(const std::vector< Vector<float,2> > &, const Color<unsigned int> &);
template void GridRender<float,2>::renderCellWithColor<unsigned char>(const std::vector< Vector<float,2> > &, const Color<unsigned char> &);

template void GridRender<float,3>::renderCellWithColor<signed char>(const std::vector< Vector<float,3> > &, const std::vector< Color<signed char> > &);
template void GridRender<float,3>::renderCellWithColor<short>(const std::vector< Vector<float,3> > &, const std::vector< Color<short> > &);
template void GridRender<float,3>::renderCellWithColor<int>(const std::vector< Vector<float,3> > &, const std::vector< Color<int> > &);
template void GridRender<float,3>::renderCellWithColor<float>(const std::vector< Vector<float,3> > &, const std::vector< Color<float> > &);
template void GridRender<float,3>::renderCellWithColor<double>(const std::vector< Vector<float,3> > &, const std::vector< Color<double> > &);
template void GridRender<float,3>::renderCellWithColor<unsigned short>(const std::vector< Vector<float,3> > &, const std::vector< Color<unsigned short> > &);
template void GridRender<float,3>::renderCellWithColor<unsigned int>(const std::vector< Vector<float,3> > &, const std::vector< Color<unsigned int> > &);
template void GridRender<float,3>::renderCellWithColor<unsigned char>(const std::vector< Vector<float,3> > &, const std::vector< Color<unsigned char> > &);

template void GridRender<float,2>::renderCellWithColor<signed char>(const std::vector< Vector<float,2> > &, const std::vector< Color<signed char> > &);
template void GridRender<float,2>::renderCellWithColor<short>(const std::vector< Vector<float,2> > &, const std::vector< Color<short> > &);
template void GridRender<float,2>::renderCellWithColor<int>(const std::vector< Vector<float,2> > &, const std::vector< Color<int> > &);
template void GridRender<float,2>::renderCellWithColor<float>(const std::vector< Vector<float,2> > &, const std::vector< Color<float> > &);
template void GridRender<float,2>::renderCellWithColor<double>(const std::vector< Vector<float,2> > &, const std::vector< Color<double> > &);
template void GridRender<float,2>::renderCellWithColor<unsigned short>(const std::vector< Vector<float,2> > &, const std::vector< Color<unsigned short> > &);
template void GridRender<float,2>::renderCellWithColor<unsigned int>(const std::vector< Vector<float,2> > &, const std::vector< Color<unsigned int> >&);
template void GridRender<float,2>::renderCellWithColor<unsigned char>(const std::vector< Vector<float,2> > &, const std::vector< Color<unsigned char> > &);

// renderNodeWithColor
template void GridRender<float,3>::renderNodeWithColor<signed char>(const std::vector< Vector<float,3> > &, const Color<signed char> &);
template void GridRender<float,3>::renderNodeWithColor<short>(const std::vector< Vector<float,3> > &, const Color<short> &);
template void GridRender<float,3>::renderNodeWithColor<int>(const std::vector< Vector<float,3> > &, const Color<int> &);
template void GridRender<float,3>::renderNodeWithColor<float>(const std::vector< Vector<float,3> > &, const Color<float> &);
template void GridRender<float,3>::renderNodeWithColor<double>(const std::vector< Vector<float,3> > &, const Color<double> &);
template void GridRender<float,3>::renderNodeWithColor<unsigned short>(const std::vector< Vector<float,3> > &, const Color<unsigned short> &);
template void GridRender<float,3>::renderNodeWithColor<unsigned int>(const std::vector< Vector<float,3> > &, const Color<unsigned int> &);
template void GridRender<float,3>::renderNodeWithColor<unsigned char>(const std::vector< Vector<float,3> > &, const Color<unsigned char> &);

template void GridRender<float,2>::renderNodeWithColor<signed char>(const std::vector< Vector<float,2> > &, const Color<signed char> &);
template void GridRender<float,2>::renderNodeWithColor<short>(const std::vector< Vector<float,2> > &, const Color<short> &);
template void GridRender<float,2>::renderNodeWithColor<int>(const std::vector< Vector<float,2> > &, const Color<int> &);
template void GridRender<float,2>::renderNodeWithColor<float>(const std::vector< Vector<float,2> > &, const Color<float> &);
template void GridRender<float,2>::renderNodeWithColor<double>(const std::vector< Vector<float,2> > &, const Color<double> &);
template void GridRender<float,2>::renderNodeWithColor<unsigned short>(const std::vector< Vector<float,2> > &, const Color<unsigned short> &);
template void GridRender<float,2>::renderNodeWithColor<unsigned int>(const std::vector< Vector<float,2> > &, const Color<unsigned int> &);
template void GridRender<float,2>::renderNodeWithColor<unsigned char>(const std::vector< Vector<float,2> > &, const Color<unsigned char> &);

template void GridRender<float,3>::renderNodeWithColor<signed char>(const std::vector< Vector<float,3> > &, const std::vector< Color<signed char> > &);
template void GridRender<float,3>::renderNodeWithColor<short>(const std::vector< Vector<float,3> > &, const std::vector< Color<short> > &);
template void GridRender<float,3>::renderNodeWithColor<int>(const std::vector< Vector<float,3> > &, const std::vector< Color<int> > &);
template void GridRender<float,3>::renderNodeWithColor<float>(const std::vector< Vector<float,3> > &, const std::vector< Color<float> > &);
template void GridRender<float,3>::renderNodeWithColor<double>(const std::vector< Vector<float,3> > &, const std::vector< Color<double> > &);
template void GridRender<float,3>::renderNodeWithColor<unsigned short>(const std::vector< Vector<float,3> > &, const std::vector< Color<unsigned short> > &);
template void GridRender<float,3>::renderNodeWithColor<unsigned int>(const std::vector< Vector<float,3> > &, const std::vector< Color<unsigned int> > &);
template void GridRender<float,3>::renderNodeWithColor<unsigned char>(const std::vector< Vector<float,3> > &, const std::vector< Color<unsigned char> > &);

template void GridRender<float,2>::renderNodeWithColor<signed char>(const std::vector< Vector<float,2> > &, const std::vector< Color<signed char> > &);
template void GridRender<float,2>::renderNodeWithColor<short>(const std::vector< Vector<float,2> > &, const std::vector< Color<short> > &);
template void GridRender<float,2>::renderNodeWithColor<int>(const std::vector< Vector<float,2> > &, const std::vector< Color<int> > &);
template void GridRender<float,2>::renderNodeWithColor<float>(const std::vector< Vector<float,2> > &, const std::vector< Color<float> > &);
template void GridRender<float,2>::renderNodeWithColor<double>(const std::vector< Vector<float,2> > &, const std::vector< Color<double> > &);
template void GridRender<float,2>::renderNodeWithColor<unsigned short>(const std::vector< Vector<float,2> > &, const std::vector< Color<unsigned short> > &);
template void GridRender<float,2>::renderNodeWithColor<unsigned int>(const std::vector< Vector<float,2> > &, const std::vector< Color<unsigned int> >&);
template void GridRender<float,2>::renderNodeWithColor<unsigned char>(const std::vector< Vector<float,2> > &, const std::vector< Color<unsigned char> > &);


} // end of namespace Physika