/*
 * @file grid_iterator.h 
 * @brief iterator for 2D/3D uniform grid
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

#ifndef PHYSIKA_GEOMETRY_CARTESIAN_GRID_GRID_ITERATOR_H_
#define PHYSIKA_GEOMETRY_CARTESIAN_GRID_GRID_ITERATOR_H_

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

template <typename Scalar,int Dim> class Grid;

/*
 * GridIteratorBase: abstract class, provide common interface/data
 * for 2D/3D uniform grid iterator 
 */
template <typename Scalar,int Dim>
class GridIteratorBase
{
protected:
    GridIteratorBase();
    virtual ~GridIteratorBase()=0;
    GridIteratorBase(const GridIteratorBase<Scalar,Dim> &iterator);
    GridIteratorBase<Scalar,Dim>& operator= (const GridIteratorBase<Scalar,Dim> &iterator);
    bool operator== (const GridIteratorBase<Scalar,Dim> &iterator) const;
    bool operator!= (const GridIteratorBase<Scalar,Dim> &iterator) const;
protected: //helper methods
    //given the high-dimensional index and resolution in each dimension, return the flat version of index
    //if given index is invalid(overflow/underflow the resolution), return -1
    int flatIndex(const Vector<int,Dim> &index, const Vector<int,Dim> &dimension) const;
    //from flat version of index and resolution in each dimension, return the high-dimensional index
    //if given flat index  is invalid(overflow/underflow the 1D resolution), return index of entries -1
    Vector<int,Dim> indexFromFlat(int flat_index, const Vector<int,Dim> &dimension) const;
    //clamp index_ variable of the iterator with given resolution in each dimension
    //if index overflow/underflow, set all its entry to -1 (invalid)
    void clampIndex(const Vector<int,Dim> &dimension);
protected:
    Vector<int,Dim> index_;
    const Grid<Scalar,Dim> *grid_;
};

/*
 * GridNodeIterator: iterator of grid node 
 *                   inherit common interface/data from GridIteratorBase
 *                   the nodes are iterated in this order: 2D, y->x; 3D, z->y->x
 *                   2D example(2X2 grid): [0][0],[0][1],[0][2],[1][0]...
 */
template <typename Scalar,int Dim>
class GridNodeIterator: public GridIteratorBase<Scalar,Dim>
{
public:
    GridNodeIterator();
    ~GridNodeIterator();
    GridNodeIterator(const GridNodeIterator<Scalar,Dim> &iterator);
    GridNodeIterator<Scalar,Dim>& operator= (const GridNodeIterator<Scalar,Dim> &iterator);
    bool operator== (const GridNodeIterator<Scalar,Dim> &iterator) const;
    bool operator!= (const GridNodeIterator<Scalar,Dim> &iterator) const;
    GridNodeIterator<Scalar,Dim>& operator++ ();
    GridNodeIterator<Scalar,Dim>& operator-- ();
    GridNodeIterator<Scalar,Dim> operator++ (int);
    GridNodeIterator<Scalar,Dim> operator-- (int);
    GridNodeIterator<Scalar,Dim> operator+ (int stride) const;
    GridNodeIterator<Scalar,Dim> operator- (int stride) const;
    const Vector<int,Dim>& nodeIndex() const;
protected:
    friend class Grid<Scalar,Dim>;
};

/*
 * GridCellIterator: iterator of grid cell 
 *                   inherit common interface/data from GridIteratorBase
 *                   the cells are iterated in this order: 2D, y->x; 3D, z->y->x
 *                   2D example(2X2 grid): [0][0],[0][1],[1][0]...
 */
template <typename Scalar,int Dim>
class GridCellIterator: public GridIteratorBase<Scalar,Dim>
{
public:
    GridCellIterator();
    ~GridCellIterator();
    GridCellIterator(const GridCellIterator<Scalar,Dim> &iterator);
    GridCellIterator<Scalar,Dim>& operator= (const GridCellIterator<Scalar,Dim> &iterator);
    bool operator== (const GridCellIterator<Scalar,Dim> &iterator) const;
    bool operator!= (const GridCellIterator<Scalar,Dim> &iterator) const;
    GridCellIterator<Scalar,Dim>& operator++ ();
    GridCellIterator<Scalar,Dim>& operator-- ();
    GridCellIterator<Scalar,Dim> operator++ (int);
    GridCellIterator<Scalar,Dim> operator-- (int);
    GridCellIterator<Scalar,Dim> operator+ (int stride) const;
    GridCellIterator<Scalar,Dim> operator- (int stride) const;
    const Vector<int,Dim>& cellIndex() const;
protected:
    friend class Grid<Scalar,Dim>;
};

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_CARTESIAN_GRID_GRID_ITERATOR_H_
