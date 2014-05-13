/*
 * @file grid.h 
 * @brief definition of uniform grid, 2D/3D
 *        iterator of the grid is defined as well
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_GEOMETRY_CARTESIAN_GRID_GRID_H_
#define PHYSIKA_GEOMETRY_CARTESIAN_GRID_GRID_H_

#include "Physika_Core/Range/range.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Cartesian_Grid/grid_iterator.h"

namespace Physika{

/*
 * The NODE, CELL of the grid can be visited either through index or iterator.
 */

/*
 * GridBase: abstract class, provide common interface/data for 2D/3D grid
 */
template <typename Scalar,int Dim>
class GridBase
{
public:
    GridBase();
    virtual ~GridBase()=0;
    GridBase(const Range<Scalar,Dim> &domain, int cell_num); //same cell size along each dimension
    GridBase(const Range<Scalar,Dim> &domain, const Vector<int,Dim> &cell_num);
    GridBase(const GridBase<Scalar,Dim> &grid);
    const Range<Scalar,Dim>& domain() const;
    const Vector<Scalar,Dim>& dX() const;
    const Vector<Scalar,Dim>& minCorner() const;
    const Vector<Scalar,Dim>& maxCorner() const;
    Scalar minEdgeLength() const;
    Scalar maxEdgeLength() const;
    const Vector<int,Dim>& cellNum() const;
    Vector<int,Dim> nodeNum() const;
    Scalar cellSize() const; //2d: area; 3d: volume;
    Vector<Scalar,Dim> node(const Vector<int,Dim> &index) const;
    Vector<Scalar,Dim> cellCenter(const Vector<int,Dim> &index) const;
    //modifiers
    void setCellNum(int cell_num);  //same cell number along each dimension
    void setCellNum(const Vector<int,Dim> &cell_num);
    void setNodeNum(int node_num);
    void setNodeNum(const Vector<int,Dim> &node_num);
    void setDomain(const Range<Scalar,Dim> &domain);
protected:
    GridBase<Scalar,Dim>& operator= (const GridBase<Scalar,Dim> &grid);
    bool operator== (const GridBase<Scalar,Dim> &grid) const;
protected:
    Range<Scalar,Dim> domain_;
    Vector<Scalar,Dim> dx_;
    Vector<int,Dim> cell_num_;
};

/*
 * Grid: inherit common interface from GridBase
 *       use partial specialization of template to provide interface specific to 2D&&3D
 */
template <typename Scalar, int Dim>
class Grid: public GridBase<Scalar,Dim>
{
};

template <typename Scalar>
class Grid<Scalar,2>: public GridBase<Scalar,2>
{
public:
    Grid():GridBase<Scalar,2>(){}
    ~Grid(){}
    Grid(const Range<Scalar,2> &domain, int cell_num):GridBase<Scalar,2>(domain,cell_num){}
    Grid(const Range<Scalar,2> &domain, const Vector<int,2> &cell_num):GridBase<Scalar,2>(domain,cell_num){}
    Grid(const Grid<Scalar,2> &grid):GridBase<Scalar,2>(grid){}
    Grid<Scalar,2>& operator= (const Grid<Scalar,2> &grid){GridBase<Scalar,2>::operator=(grid);return *this;}
    bool operator== (const Grid<Scalar,2> &grid){return GridBase<Scalar,2>::operator==(grid);}
    Vector<Scalar,2> node(int i, int j) const;
    Vector<Scalar,2> cellCenter(int i, int j) const;
    //avoid hiding node() and cellCenter() methods in GridBase
    using GridBase<Scalar,2>::node;
    using GridBase<Scalar,2>::cellCenter;

    typedef GridNodeIterator<Scalar,2> NodeIterator;
    typedef GridCellIterator<Scalar,2> CellIterator;
    NodeIterator nodeBegin() const;
    NodeIterator nodeEnd() const;
    CellIterator cellBegin() const;
    CellIterator cellEnd() const;
};

template <typename Scalar>
class Grid<Scalar,3>: public GridBase<Scalar,3>
{
public:
    Grid():GridBase<Scalar,3>(){}
    ~Grid(){}
    Grid(const Range<Scalar,3> &domain, int cell_num):GridBase<Scalar,3>(domain,cell_num){}
    Grid(const Range<Scalar,3> &domain, const Vector<int,3> &cell_num):GridBase<Scalar,3>(domain,cell_num){}
    Grid(const Grid<Scalar,3> &grid):GridBase<Scalar,3>(grid){}
    Grid<Scalar,3>& operator= (const Grid<Scalar,3> &grid){GridBase<Scalar,3>::operator=(grid);return *this;}
    bool operator== (const Grid<Scalar,3> &grid){return GridBase<Scalar,3>::operator==(grid);}
    Vector<Scalar,3> node(int i, int j, int k) const;
    Vector<Scalar,3> cellCenter(int i, int j, int k) const;
    //avoid hiding node() and cellCenter() methods in GridBase
    using GridBase<Scalar,3>::node;
    using GridBase<Scalar,3>::cellCenter;

    typedef GridNodeIterator<Scalar,3> NodeIterator;
    typedef GridCellIterator<Scalar,3> CellIterator;
    NodeIterator nodeBegin() const;
    NodeIterator nodeEnd() const;
    CellIterator cellBegin() const;
    CellIterator cellEnd() const;
};

}  //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_CARTESIAN_GRID_GRID_H_
