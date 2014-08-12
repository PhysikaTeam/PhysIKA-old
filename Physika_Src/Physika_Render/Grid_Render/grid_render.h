/*
 * @file grid_render.h 
 * @Brief render of grid.
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

#ifndef PHYSIKA_RENDER_GRID_RENDER_GRID_RENDER_H_
#define PHYSIKA_RENDER_GRID_RENDER_GRID_RENDER_H_

#include <utility>
#include <vector>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Render/Render_Base/render_base.h"

namespace Physika{

template <typename Scalar, int Dim> class Grid;
template <typename Scalar> class Color;

/*
 * Scalar can be float and double
 *
 */

template <typename Scalar, int Dim>
class GridRender: public RenderBase
{
public:
    // constrctor and destructor
    GridRender();
    explicit GridRender(Grid<Scalar,Dim> * grid);
    ~GridRender();

    // getter and setter
    const Grid<Scalar,Dim> * grid() const;
    void setGrid(Grid<Scalar,Dim> * grid);

    //whenever the mesh is modified, synchronize() must be called to update the render
    void synchronize();

    virtual void render();
    template <typename ColorType>
    void renderNodeWithColor(const std::vector< Vector<unsigned int,Dim> > & node_vec, const Color<ColorType> &color);
    template <typename ColorType>
    void renderNodeWithColor(const std::vector< Vector<unsigned int,Dim> > & node_vec, const std::vector< Color<ColorType> > &color);

    template <typename ColorType>
    void renderCellWithColor(const std::vector< Vector<unsigned int,Dim> > & cell_vec, const Color<ColorType> &color);
    template <typename ColorType>
    void renderCellWithColor(const std::vector< Vector<unsigned int,Dim> > & cell_vec, const std::vector< Color<ColorType> > &color);

    virtual void printInfo() const;

protected:
    void renderCell(const Vector<unsigned int, Dim> & cell_idx);
    Grid<Scalar,Dim> * grid_;
    unsigned int display_list_id_;
};

} // end of namespace Physika
#endif // PHYSIKA_RENDER_GRID_RENDER_GRID_RENDER_H_