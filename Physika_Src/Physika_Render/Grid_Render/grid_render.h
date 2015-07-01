/*
 * @file grid_render.h 
 * @Brief render of grid.
 * @author Wei Chen, Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_RENDER_GRID_RENDER_GRID_RENDER_H_
#define PHYSIKA_RENDER_GRID_RENDER_GRID_RENDER_H_

#include <vector>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Render/Color/color.h"
#include "Physika_Render/Render_Base/render_base.h"

namespace Physika{

template <typename Scalar, int Dim> class Grid;
template <typename Scalar> class Color;

/*
 * Scalar can be those scalar types which grid are defind for, i.e.,
 * unsigned short, unsigned int, unsigned long, unsigned long long,
 * signed char, short, int, long, long long, float, double, long double
 *
 * Dim can be 2 and 3
 */

template <typename Scalar, int Dim>
class GridRender: public RenderBase
{
public:
    // constrctor and destructor
    GridRender();
    explicit GridRender(const Grid<Scalar,Dim> *grid);
    ~GridRender();

    // getter and setter
    const Grid<Scalar,Dim>* grid() const;
    void setGrid(const Grid<Scalar,Dim> *grid);

    //whenever the grid is modified, synchronize() must be called to update the render
    void synchronize();

	// render in default mode
    virtual void render();

    // set the color used to rendered grid
    template <typename ColorType>
    void setGridColor(const Color<ColorType> &color);

	// note: the highlighted node will be rendered 3 times larger. 
    template <typename ColorType>
    void renderNodeWithColor(const std::vector< Vector<unsigned int,Dim> > & node_vec, const Color<ColorType> &color);
    template <typename ColorType>
    void renderNodeWithColor(const std::vector< Vector<unsigned int,Dim> > & node_vec, const std::vector< Color<ColorType> > &color);

	// note: the edges of cell will be rendered 2 times wider.
    template <typename ColorType>
    void renderCellWithColor(const std::vector< Vector<unsigned int,Dim> > & cell_vec, const Color<ColorType> &color);
    template <typename ColorType>
    void renderCellWithColor(const std::vector< Vector<unsigned int,Dim> > & cell_vec, const std::vector< Color<ColorType> > &color);

    virtual void printInfo() const;

protected:
    void renderCell(const Vector<unsigned int, Dim> & cell_idx);
    const Grid<Scalar,Dim> * grid_;
    unsigned int display_list_id_;
    Color<double> grid_color_;  //grid render color, stored as double type
};

} // end of namespace Physika

#include "Physika_Render/Grid_Render/grid_render-inl.h"

#endif // PHYSIKA_RENDER_GRID_RENDER_GRID_RENDER_H_
