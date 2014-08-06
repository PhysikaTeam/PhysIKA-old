/*
 * @file grid.cpp 
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

#include <limits>
#include <cstdlib>
#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Geometry/Cartesian_Grids/grid.h"

namespace Physika{

template <typename Scalar,int Dim>
GridBase<Scalar,Dim>::GridBase()
{
}

template <typename Scalar,int Dim>
GridBase<Scalar,Dim>::~GridBase()
{
}

template <typename Scalar,int Dim>
GridBase<Scalar,Dim>::GridBase(const Range<Scalar,Dim> &domain, unsigned int cell_num)
    :domain_(domain)
{
    if(cell_num == 0)
    {
        std::cerr<<"Error: grid cell number must be greator than 1, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    cell_num_ = Vector<unsigned int,Dim>(cell_num);
    dx_ = domain_.edgeLengths()/cell_num;
}

template <typename Scalar,int Dim>
GridBase<Scalar,Dim>::GridBase(const Range<Scalar,Dim> &domain, const Vector<unsigned int,Dim> &cell_num)
    :domain_(domain)
{
    for(unsigned int i = 0; i < Dim; ++i)
    {
        if(cell_num[i] == 0)
        {
            std::cerr<<"Error: grid cell number must be greator than 1, program abort!\n";
            std::exit(EXIT_FAILURE);
        }
    }
    cell_num_ = cell_num;
    Vector<Scalar,Dim> domain_size = domain_.edgeLengths();
    for(unsigned int i = 0; i < Dim; ++i)
        dx_[i] = domain_size[i]/cell_num_[i];
}

template <typename Scalar,int Dim>
GridBase<Scalar,Dim>::GridBase(const GridBase<Scalar,Dim> &grid)
    :domain_(grid.domain_),dx_(grid.dx_),cell_num_(grid.cell_num_)
{
}

template <typename Scalar,int Dim>
GridBase<Scalar,Dim>& GridBase<Scalar,Dim>::operator= (const GridBase<Scalar,Dim> &grid)
{
    domain_ = grid.domain_;
    dx_ = grid.dx_;
    cell_num_ = grid.cell_num_;
    return *this;
}

template <typename Scalar,int Dim>
bool GridBase<Scalar,Dim>::operator== (const GridBase<Scalar,Dim> &grid) const
{
    return (domain_==grid.domain_)&&(dx_==grid.dx_)&&(cell_num_==grid.cell_num_);
}

template <typename Scalar,int Dim>
Range<Scalar,Dim> GridBase<Scalar,Dim>::domain() const
{
    return domain_;
}

template <typename Scalar,int Dim>
Vector<Scalar,Dim> GridBase<Scalar,Dim>::dX() const
{
    return dx_;
}

template <typename Scalar,int Dim>
Vector<Scalar,Dim> GridBase<Scalar,Dim>::minCorner() const
{
    return domain_.minCorner();
}

template <typename Scalar,int Dim>
Vector<Scalar,Dim> GridBase<Scalar,Dim>::maxCorner() const
{
    return domain_.maxCorner();
}

template <typename Scalar,int Dim>
Scalar GridBase<Scalar,Dim>::minEdgeLength() const
{
    Scalar min_length = std::numeric_limits<Scalar>::max();
    for(unsigned int i = 0; i < Dim; ++i)
        if(dx_[i]<min_length)
            min_length = dx_[i];
    return min_length;
}

template <typename Scalar,int Dim>
Scalar GridBase<Scalar,Dim>::maxEdgeLength() const
{
    Scalar max_length = std::numeric_limits<Scalar>::min();
    for(unsigned int i = 0; i < Dim; ++i)
        if(dx_[i]>max_length)
            max_length = dx_[i];
    return max_length;
}

template <typename Scalar,int Dim>
Vector<unsigned int,Dim> GridBase<Scalar,Dim>::cellNum() const
{
    return cell_num_;
}

template <typename Scalar,int Dim>
Vector<unsigned int,Dim> GridBase<Scalar,Dim>::nodeNum() const
{
    return cell_num_+Vector<unsigned int,Dim>(1);
}

template <typename Scalar,int Dim>
Scalar GridBase<Scalar,Dim>::cellSize() const
{
    Scalar size = 1.0;
    for(unsigned int i = 0; i < Dim; ++i)
        size *= dx_[i];
    return size;
}

template <typename Scalar,int Dim>
Vector<Scalar,Dim> GridBase<Scalar,Dim>::node(const Vector<unsigned int,Dim> &node_idx) const
{
    for(unsigned int i = 0; i < Dim; ++i)
    {
        if(node_idx[i]>cell_num_[i])
        {
            std::cerr<<"Error: Grid node index out of range, program abort!\n";
            std::exit(EXIT_FAILURE);
        }
    }
    Vector<Scalar,Dim> bias;
    for(unsigned int i = 0; i < Dim; ++i)
        bias[i] = node_idx[i]*dx_[i];
    return domain_.minCorner()+bias;
}

template <typename Scalar,int Dim>
Vector<Scalar,Dim> GridBase<Scalar,Dim>::cellCenter(const Vector<unsigned int,Dim> &cell_idx) const
{
    for(unsigned int i = 0; i < Dim; ++i)
    {
        if(cell_idx[i]>=cell_num_[i])
        {
            std::cerr<<"Error: Grid cell index out of range, program abort!\n";
            std::exit(EXIT_FAILURE);
        }
    }
    Vector<Scalar,Dim> cor_node = node(cell_idx);
    return cor_node+0.5*dx_;
}

template <typename Scalar,int Dim>
Vector<Scalar,Dim> GridBase<Scalar,Dim>::cellMinCornerNode(const Vector<unsigned int,Dim> &cell_idx) const
{
    for(unsigned int i = 0; i < Dim; ++i)
    {
        if(cell_idx[i]>=cell_num_[i])
        {
            std::cerr<<"Error: Grid cell index out of range, program abort!\n";
            std::exit(EXIT_FAILURE);
        }
    }
    return node(cell_idx);
}

template <typename Scalar,int Dim>
Vector<Scalar,Dim> GridBase<Scalar,Dim>::cellMaxCornerNode(const Vector<unsigned int,Dim> &cell_idx) const
{
    Vector<Scalar,Dim> min_node = cellMinCornerNode(cell_idx);
    return min_node+dx_;
}

template <typename Scalar,int Dim>
Vector<unsigned int,Dim> GridBase<Scalar,Dim>::cellIndex(const Vector<Scalar,Dim> &position) const
{
    Vector<unsigned int,Dim> cell_idx;
    bool out_range = false;
    Vector<Scalar,Dim> bias = position - domain_.minCorner();
    for(unsigned int i = 0; i < Dim; ++i)
    {
        PHYSIKA_ASSERT(dx_[i]>0);
        if(bias[i] < 0)
        {
            out_range = true;
            bias[i] = 0;
        }
        cell_idx[i] = static_cast<unsigned int>(bias[i]/dx_[i]);
        if(cell_idx[i] >= cell_num_[i])
        {
            out_range = true;
            cell_idx[i] = cell_num_[i] -1;
        }
    }
    if(out_range)
        std::cerr<<"Warning: Point out of grid range, clamped to closest cell!\n";
    return cell_idx;
}

template <typename Scalar,int Dim>
void GridBase<Scalar,Dim>::cellIndexAndInterpolationWeight(const Vector<Scalar,Dim> &position,
                                                           Vector<unsigned int,Dim> &cell_idx, Vector<Scalar,Dim> &weight) const
{
    bool out_range = false;
    Vector<Scalar,Dim> bias = position - domain_.minCorner();
    for(unsigned int i = 0; i < Dim; ++i)
    {
        PHYSIKA_ASSERT(dx_[i]>0);
        if(bias[i] < 0)
        {
            out_range = true;
            bias[i] = 0;
        }
        cell_idx[i] = static_cast<unsigned int>(bias[i]/dx_[i]);
        if(cell_idx[i] >= cell_num_[i])
        {
            out_range = true;
            cell_idx[i] = cell_num_[i] -1;
        }
        weight[i] = bias[i]/dx_[i] - cell_idx[i];
    }
    if(out_range)
        std::cerr<<"Warning: Point out of grid range, clamped to closest cell!\n";
}

template <typename Scalar,int Dim>
void GridBase<Scalar,Dim>::setCellNum(unsigned int cell_num)
{
    cell_num_ = Vector<unsigned int,Dim>(cell_num);
    dx_ = domain_.edgeLengths()/cell_num;
}

template <typename Scalar,int Dim>
void GridBase<Scalar,Dim>::setCellNum(const Vector<unsigned int,Dim> &cell_num)
{
    for(unsigned int i = 0; i < Dim; ++i)
    {
        if(cell_num[i]<=0)
        {
            std::cerr<<"Error: Cell number of a grid must be greater than zero, program abort!\n";
            std::exit(EXIT_FAILURE);
        }
    }
    cell_num_ = cell_num;
    Vector<Scalar,Dim> domain_size = domain_.edgeLengths();
    for(unsigned int i = 0; i < Dim; ++i)
        dx_[i] = domain_size[i]/cell_num_[i];
}

template <typename Scalar,int Dim>
void GridBase<Scalar,Dim>::setNodeNum(unsigned int node_num)
{
    if(node_num<2)
    {
        std::cerr<<"Error: Node number of a grid must be greater than 1, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    cell_num_ = Vector<unsigned int,Dim>(node_num-1);
    dx_ = domain_.edgeLengths()/(node_num-1);
}

template <typename Scalar,int Dim>
void GridBase<Scalar,Dim>::setNodeNum(const Vector<unsigned int,Dim> &node_num)
{
    for(unsigned int i = 0; i < Dim; ++i)
        if(node_num[i]<2)
        {
            std::cerr<<"Error: Node number of a grid must be greater than 1, program abort!\n";
            std::exit(EXIT_FAILURE);
        }
    cell_num_ = node_num-Vector<unsigned int,Dim>(1);
    Vector<Scalar,Dim> domain_size = domain_.edgeLengths();
    for(unsigned int i = 0; i < Dim; ++i)
        dx_[i] = domain_size[i]/cell_num_[i];
}

template <typename Scalar,int Dim>
void GridBase<Scalar,Dim>::setDomain(const Range<Scalar,Dim> &domain)
{
    domain_ = domain;
    Vector<Scalar,Dim> domain_size = domain_.edgeLengths();
    for(unsigned int i = 0; i < Dim; ++i)
        dx_[i] = domain_size[i]/cell_num_[i];
}

template <typename Scalar>
Grid<Scalar,2>::Grid(const Range<Scalar,2> &domain, unsigned int cell_num)
 :GridBase<Scalar,2>(domain,cell_num)
{
}

template <typename Scalar>
Grid<Scalar,2>::Grid(const Range<Scalar,2> &domain, const Vector<unsigned int,2> &cell_num)
 :GridBase<Scalar,2>(domain,cell_num)
{
}

template <typename Scalar>
Grid<Scalar,2>::Grid(const Grid<Scalar,2> &grid)
 :GridBase<Scalar,2>(grid)
{
}

template <typename Scalar>
Grid<Scalar,2>& Grid<Scalar,2>::operator= (const Grid<Scalar,2> &grid)
{
    GridBase<Scalar,2>::operator=(grid);
    return *this;
}

template <typename Scalar>
bool Grid<Scalar,2>::operator== (const Grid<Scalar,2> &grid) const
{
    return GridBase<Scalar,2>::operator==(grid);
}

template <typename Scalar>
Vector<Scalar,2> Grid<Scalar,2>::node(unsigned int i, unsigned int j) const
{
    Vector<unsigned int,2> index(i,j);
    return GridBase<Scalar,2>::node(index);  //still wonder why this->node(index) doesn't work
}

template <typename Scalar>
Vector<Scalar,2> Grid<Scalar,2>::cellCenter(unsigned int i, unsigned int j) const
{
    Vector<unsigned int,2> index(i,j);
    return GridBase<Scalar,2>::cellCenter(index);
}

template <typename Scalar>
Vector<Scalar,2> Grid<Scalar,2>::cellMinCornerNode(unsigned int i, unsigned int j) const
{
    Vector<unsigned int,2> index(i,j);
    return GridBase<Scalar,2>::cellMinCornerNode(index);
}

template <typename Scalar>
Vector<Scalar,2> Grid<Scalar,2>::cellMaxCornerNode(unsigned int i, unsigned int j) const
{
    Vector<Scalar,2> min_node = cellMinCornerNode(i,j);
    return min_node + this->dx_;
}

template <typename Scalar>
typename Grid<Scalar,2>::NodeIterator Grid<Scalar,2>::nodeBegin() const
{
    Grid<Scalar,2>::NodeIterator iterator;
    iterator.index_ = Vector<unsigned int,2>(0);
    iterator.grid_ = this;
    return iterator;
}

template <typename Scalar>
typename Grid<Scalar,2>::NodeIterator Grid<Scalar,2>::nodeEnd() const
{
    Grid<Scalar,2>::NodeIterator iterator;
    Vector<unsigned int,2> node_num = GridBase<Scalar,2>::nodeNum();
    iterator.index_ = Vector<unsigned int,2>(node_num[0],0);
    iterator.grid_ = this;
    return iterator;
}

template <typename Scalar>
typename Grid<Scalar,2>::CellIterator Grid<Scalar,2>::cellBegin() const
{
    Grid<Scalar,2>::CellIterator iterator;
    iterator.index_ = Vector<unsigned int,2>(0);
    iterator.grid_ = this;
    return iterator;
}

template <typename Scalar>
typename Grid<Scalar,2>::CellIterator Grid<Scalar,2>::cellEnd() const
{
    Grid<Scalar,2>::CellIterator iterator;
    const Vector<unsigned int,2> &cell_num = this->cell_num_;
    iterator.index_ = Vector<unsigned int,2>(cell_num[0],0);
    iterator.grid_ = this;
    return iterator;
}


template <typename Scalar>
Grid<Scalar,3>::Grid(const Range<Scalar,3> &domain, unsigned int cell_num)
 :GridBase<Scalar,3>(domain,cell_num)
{
}

template <typename Scalar>
Grid<Scalar,3>::Grid(const Range<Scalar,3> &domain, const Vector<unsigned int,3> &cell_num)
 :GridBase<Scalar,3>(domain,cell_num)
{
}

template <typename Scalar>
Grid<Scalar,3>::Grid(const Grid<Scalar,3> &grid)
 :GridBase<Scalar,3>(grid)
{
}

template <typename Scalar>
Grid<Scalar,3>& Grid<Scalar,3>::operator= (const Grid<Scalar,3> &grid)
{
    GridBase<Scalar,3>::operator=(grid);
    return *this;
}

template <typename Scalar>
bool Grid<Scalar,3>::operator== (const Grid<Scalar,3> &grid) const
{
    return GridBase<Scalar,3>::operator==(grid);
}

template <typename Scalar>
Vector<Scalar,3> Grid<Scalar,3>::node(unsigned int i, unsigned int j, unsigned int k) const
{
    Vector<unsigned int,3> index(i,j,k);
    return GridBase<Scalar,3>::node(index);
}

template <typename Scalar>
Vector<Scalar,3> Grid<Scalar,3>::cellCenter(unsigned int i, unsigned int j, unsigned int k) const
{
    Vector<unsigned int,3> index(i,j,k);
    return GridBase<Scalar,3>::cellCenter(index);
}

template <typename Scalar>
Vector<Scalar,3> Grid<Scalar,3>::cellMinCornerNode(unsigned int i, unsigned int j, unsigned int k) const
{
    Vector<unsigned int,3> index(i,j,k);
    return GridBase<Scalar,3>::cellMinCornerNode(index);
}

template <typename Scalar>
Vector<Scalar,3> Grid<Scalar,3>::cellMaxCornerNode(unsigned int i, unsigned int j, unsigned int k) const
{
    Vector<Scalar,3> min_node = cellMinCornerNode(i,j,k);
    return min_node + this->dx_;
}

template <typename Scalar>
typename Grid<Scalar,3>::NodeIterator Grid<Scalar,3>::nodeBegin() const
{
    Grid<Scalar,3>::NodeIterator iterator;
    iterator.index_ = Vector<unsigned int,3>(0);
    iterator.grid_ = this;
    return iterator;
}

template <typename Scalar>
typename Grid<Scalar,3>::NodeIterator Grid<Scalar,3>::nodeEnd() const
{
    Grid<Scalar,3>::NodeIterator iterator;
    Vector<unsigned int,3> node_num = GridBase<Scalar,3>::nodeNum();
    iterator.index_ = Vector<unsigned int,3>(node_num[0],0,0);
    iterator.grid_ = this;
    return iterator;
}

template <typename Scalar>
typename Grid<Scalar,3>::CellIterator Grid<Scalar,3>::cellBegin() const
{
    Grid<Scalar,3>::CellIterator iterator;
    iterator.index_ = Vector<unsigned int,3>(0);
    iterator.grid_ = this;
    return iterator;
}

template <typename Scalar>
typename Grid<Scalar,3>::CellIterator Grid<Scalar,3>::cellEnd() const
{
    Grid<Scalar,3>::CellIterator iterator;
    const Vector<unsigned int,3> &cell_num = this->cell_num_;
    iterator.index_ = Vector<unsigned int,3>(cell_num[0],0,0);
    iterator.grid_ = this;
    return iterator;
}

//explicit instantiation
template class GridBase<unsigned char,2>;
template class GridBase<unsigned short,2>;
template class GridBase<unsigned int,2>;
template class GridBase<unsigned long,2>;
template class GridBase<unsigned long long,2>;
template class GridBase<signed char,2>;
template class GridBase<short,2>;
template class GridBase<int,2>;
template class GridBase<long,2>;
template class GridBase<long long,2>;
template class GridBase<float,2>;
template class GridBase<double,2>;
template class GridBase<long double,2>;
template class GridBase<unsigned char,3>;
template class GridBase<unsigned short,3>;
template class GridBase<unsigned int,3>;
template class GridBase<unsigned long,3>;
template class GridBase<unsigned long long,3>;
template class GridBase<signed char,3>;
template class GridBase<short,3>;
template class GridBase<int,3>;
template class GridBase<long,3>;
template class GridBase<long long,3>;
template class GridBase<float,3>;
template class GridBase<double,3>;
template class GridBase<long double,3>;

template class Grid<unsigned char,2>;
template class Grid<unsigned short,2>;
template class Grid<unsigned int,2>;
template class Grid<unsigned long,2>;
template class Grid<unsigned long long,2>;
template class Grid<signed char,2>;
template class Grid<short,2>;
template class Grid<int,2>;
template class Grid<long,2>;
template class Grid<long long,2>;
template class Grid<float,2>;
template class Grid<double,2>;
template class Grid<long double,2>;
template class Grid<unsigned char,3>;
template class Grid<unsigned short,3>;
template class Grid<unsigned int,3>;
template class Grid<unsigned long,3>;
template class Grid<unsigned long long,3>;
template class Grid<signed char,3>;
template class Grid<short,3>;
template class Grid<int,3>;
template class Grid<long,3>;
template class Grid<long long,3>;
template class Grid<float,3>;
template class Grid<double,3>;
template class Grid<long double,3>;

} //end of namespace Physika
