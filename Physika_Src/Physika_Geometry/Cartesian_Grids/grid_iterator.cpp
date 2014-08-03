/*
 * @file grid_iterator.cpp 
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

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include "Physika_Geometry/Cartesian_Grids/grid_iterator.h"
#include "Physika_Geometry/Cartesian_Grids/grid.h"

namespace Physika{

template <typename Scalar,int Dim>
GridIteratorBase<Scalar,Dim>::GridIteratorBase()
    :index_(0),grid_(NULL)
{
}

template <typename Scalar,int Dim>
GridIteratorBase<Scalar,Dim>::~GridIteratorBase()
{
}

template <typename Scalar,int Dim>
GridIteratorBase<Scalar,Dim>::GridIteratorBase(const GridIteratorBase<Scalar,Dim> &iterator)
    :index_(iterator.index_),grid_(iterator.grid_)
{
}

template <typename Scalar,int Dim>
GridIteratorBase<Scalar,Dim>& GridIteratorBase<Scalar,Dim>::operator= (const GridIteratorBase<Scalar,Dim> &iterator)
{
    index_ = iterator.index_;
    grid_ = iterator.grid_;
    return *this;
}

template <typename Scalar,int Dim>
bool GridIteratorBase<Scalar,Dim>::operator== (const GridIteratorBase<Scalar,Dim> &iterator) const
{
    if(this->grid_==NULL)
    {
        std::cerr<<"Undefined operator == for uninitialized iterator!\n";
        std::exit(EXIT_FAILURE);
    }
    return (index_==iterator.index_)&&(grid_==iterator.grid_);
}

template <typename Scalar,int Dim>
bool GridIteratorBase<Scalar,Dim>::operator!= (const GridIteratorBase<Scalar,Dim> &iterator) const
{
    if(this->grid_==NULL)
    {
        std::cerr<<"Undefined operator != for uninitialized iterator!\n";
        std::exit(EXIT_FAILURE);
    }
    return (!(index_==iterator.index_)) || (grid_!=iterator.grid_);
}

template <typename Scalar,int Dim>
unsigned int GridIteratorBase<Scalar,Dim>::flatIndex(const Vector<unsigned int,Dim> &index, const Vector<unsigned int,Dim> &dimension) const
{
    unsigned int flat_index = 0;
    Vector<unsigned int,Dim> vec = index;
    for(unsigned int i = 0; i < Dim; ++i)
    {
        for(unsigned int j = i+1; j < Dim; ++j)
            vec[i] *= dimension[j];
        flat_index += vec[i];
    }
    return flat_index;
}

template <typename Scalar,int Dim>
Vector<unsigned int,Dim> GridIteratorBase<Scalar,Dim>::indexFromFlat(unsigned int flat_index, const Vector<unsigned int,Dim> &dimension) const
{
    Vector<unsigned int,Dim> index(1);
    for(unsigned int i = 0; i < Dim; ++i)
    {
        for(unsigned int j = i+1; j < Dim; ++j)
            index[i] *= dimension[j];
        unsigned int temp = flat_index / index[i];
        flat_index = flat_index % index[i];
        index[i] = temp;
    }
    return index;
}

template <typename Scalar,int Dim>
bool GridIteratorBase<Scalar,Dim>::indexCheck(const Vector<unsigned int,Dim> &dimension) const
{
    for(unsigned int i = 0; i < Dim; ++i)
        if(index_[i]>=dimension[i])
            return false;
    return true;
}

template <typename Scalar,int Dim>
GridNodeIterator<Scalar,Dim>::GridNodeIterator()
    :GridIteratorBase<Scalar,Dim>()
{
}

template <typename Scalar,int Dim>
GridNodeIterator<Scalar,Dim>::~GridNodeIterator()
{
}

template <typename Scalar,int Dim>
GridNodeIterator<Scalar,Dim>::GridNodeIterator(const GridNodeIterator<Scalar,Dim> &iterator)
    :GridIteratorBase<Scalar,Dim>(iterator)
{
}

template <typename Scalar,int Dim>
GridNodeIterator<Scalar,Dim>& GridNodeIterator<Scalar,Dim>::operator= (const GridNodeIterator<Scalar,Dim> &iterator)
{
    GridIteratorBase<Scalar,Dim>::operator= (iterator);
    return *this;
}

template <typename Scalar,int Dim>
bool GridNodeIterator<Scalar,Dim>::operator== (const GridNodeIterator<Scalar,Dim> &iterator) const
{
    return GridIteratorBase<Scalar,Dim>::operator== (iterator);
}

template <typename Scalar,int Dim>
bool GridNodeIterator<Scalar,Dim>::operator!= (const GridNodeIterator<Scalar,Dim> &iterator) const
{
    return GridIteratorBase<Scalar,Dim>::operator!= (iterator);;
}

template <typename Scalar,int Dim>
GridNodeIterator<Scalar,Dim>& GridNodeIterator<Scalar,Dim>::operator++ ()
{
    if(this->grid_==NULL)
    {
        std::cerr<<"Undefined operator ++ for uninitialized iterator!\n";
        std::exit(EXIT_FAILURE);
    }
    Vector<unsigned int,Dim> node_num = (this->grid_)->nodeNum();
    Vector<unsigned int,Dim> &index = this->index_; //for ease of coding
    for(unsigned int i = Dim-1; i >= 0; --i)
    {
        if((index[i]==node_num[i]-1) && (i!=0))
            index[i] = 0;
        else
        {
            ++index[i];
            break;
        }
    }
    return *this;
}

template <typename Scalar,int Dim>
GridNodeIterator<Scalar,Dim>& GridNodeIterator<Scalar,Dim>::operator-- ()
{
    if(this->grid_==NULL)
    {
        std::cerr<<"Undefined operator -- for uninitialized iterator!\n";
        std::exit(EXIT_FAILURE);
    }
    Vector<unsigned int,Dim> node_num = (this->grid_)->nodeNum();
    Vector<unsigned int,Dim> &index = this->index_; //for ease of coding
    for(unsigned int i = Dim-1; i >= 0; --i)
    {
        if((index[i]==0) && (i!=0))
            index[i] = node_num[i]-1;
        else
        {
            --index[i];
            break;
        }
    }
    return *this;
}

template <typename Scalar,int Dim>
GridNodeIterator<Scalar,Dim> GridNodeIterator<Scalar,Dim>::operator++ (int)
{
    if(this->grid_==NULL)
    {
        std::cerr<<"Undefined operator ++ for uninitialized iterator!\n";
        std::exit(EXIT_FAILURE);
    }
    GridNodeIterator<Scalar,Dim> iterator(*this);
    Vector<unsigned int,Dim> node_num = (this->grid_)->nodeNum();
    Vector<unsigned int,Dim> &index = this->index_; //for ease of coding
    for(unsigned int i = Dim-1; i >= 0; --i)
    {
        if((index[i]==node_num[i]-1) && (i!=0))
            index[i] = 0;
        else
        {
            ++index[i];
            break;
        }
    }
    return iterator;
}

template <typename Scalar,int Dim>
GridNodeIterator<Scalar,Dim> GridNodeIterator<Scalar,Dim>::operator-- (int)
{
    if(this->grid_==NULL)
    {
        std::cerr<<"Undefined operator -- for uninitialized iterator!\n";
        std::exit(EXIT_FAILURE);
    }
    GridNodeIterator<Scalar,Dim> iterator(*this);
    Vector<unsigned int,Dim> node_num = (this->grid_)->nodeNum();
    Vector<unsigned int,Dim> &index = this->index_; //for ease of coding
    for(unsigned int i = Dim-1; i >= 0; --i)
    {
        if((index[i]==0) && (i!=0))
            index[i] = node_num[i]-1;
        else
        {
            --index[i];
            break;
        }
    }
    return iterator;
}

template <typename Scalar,int Dim>
GridNodeIterator<Scalar,Dim> GridNodeIterator<Scalar,Dim>::operator+ (int stride) const
{
    if(this->grid_==NULL)
    {
        std::cerr<<"Undefined operator + for uninitialized iterator!\n";
        std::exit(EXIT_FAILURE);
    }
    GridNodeIterator<Scalar,Dim> iterator(*this);
    Vector<unsigned int,Dim> node_num = (iterator.grid_)->nodeNum();
    Vector<unsigned int,Dim> &index = iterator.index_;
    unsigned int flat_index = GridIteratorBase<Scalar,Dim>::flatIndex(index,node_num);
    flat_index += stride;
    index = GridIteratorBase<Scalar,Dim>::indexFromFlat(flat_index,node_num);
    return iterator;
}

template <typename Scalar,int Dim>
GridNodeIterator<Scalar,Dim> GridNodeIterator<Scalar,Dim>::operator- (int stride) const
{
    if(this->grid_==NULL)
    {
        std::cerr<<"Undefined operator - for uninitialized iterator!\n";
        std::exit(EXIT_FAILURE);
    }
    GridNodeIterator<Scalar,Dim> iterator(*this);
    Vector<unsigned int,Dim> node_num = (iterator.grid_)->nodeNum();
    Vector<unsigned int,Dim> &index = iterator.index_;
    unsigned int flat_index = GridIteratorBase<Scalar,Dim>::flatIndex(index,node_num);
    flat_index -= stride;
    index = GridIteratorBase<Scalar,Dim>::indexFromFlat(flat_index,node_num);
    return iterator;
}

template <typename Scalar,int Dim>
Vector<unsigned int,Dim> GridNodeIterator<Scalar,Dim>::nodeIndex() const
{
    if(this->validCheck()==false)
    {
        std::cerr<<"nodeIndex(): Invalid iterator!\n";
        std::exit(EXIT_FAILURE);
    }
    return this->index_;
}

template <typename Scalar,int Dim>
bool GridNodeIterator<Scalar,Dim>::validCheck() const
{
    if(this->grid_==NULL)
        return false;
    Vector<unsigned int,Dim> node_num = (this->grid_)->nodeNum();
    if(GridIteratorBase<Scalar,Dim>::indexCheck(node_num)==false)
        return false;
    return true;
}

template <typename Scalar,int Dim>
GridCellIterator<Scalar,Dim>::GridCellIterator()
    :GridIteratorBase<Scalar,Dim>()
{
}

template <typename Scalar,int Dim>
GridCellIterator<Scalar,Dim>::~GridCellIterator()
{
}

template <typename Scalar,int Dim>
GridCellIterator<Scalar,Dim>::GridCellIterator(const GridCellIterator<Scalar,Dim> &iterator)
    :GridIteratorBase<Scalar,Dim>(iterator)
{
}

template <typename Scalar,int Dim>
GridCellIterator<Scalar,Dim>& GridCellIterator<Scalar,Dim>::operator= (const GridCellIterator<Scalar,Dim> &iterator)
{
    GridIteratorBase<Scalar,Dim>::operator= (iterator);
    return *this;
}

template <typename Scalar,int Dim>
bool GridCellIterator<Scalar,Dim>::operator== (const GridCellIterator<Scalar,Dim> &iterator) const
{
    return GridIteratorBase<Scalar,Dim>::operator== (iterator);
}

template <typename Scalar,int Dim>
bool GridCellIterator<Scalar,Dim>::operator!= (const GridCellIterator<Scalar,Dim> &iterator) const
{
    return GridIteratorBase<Scalar,Dim>::operator!= (iterator);;
}

template <typename Scalar,int Dim>
GridCellIterator<Scalar,Dim>& GridCellIterator<Scalar,Dim>::operator++ ()
{
    if(this->grid_==NULL)
    {
        std::cerr<<"Undefined operator ++ for uninitialized iterator!\n";
        std::exit(EXIT_FAILURE);
    }
    Vector<unsigned int,Dim> cell_num = (this->grid_)->cellNum();
    Vector<unsigned int,Dim> &index = this->index_; //for ease of coding
    for(unsigned int i = Dim-1; i >= 0; --i)
    {
        if((index[i]==cell_num[i]-1) && (i!=0))
            index[i] = 0;
        else
        {
            ++index[i];
            break;
        }
    }
    return *this;
}

template <typename Scalar,int Dim>
GridCellIterator<Scalar,Dim>& GridCellIterator<Scalar,Dim>::operator-- ()
{
    if(this->grid_==NULL)
    {
        std::cerr<<"Undefined operator -- for uninitialized iterator!\n";
        std::exit(EXIT_FAILURE);
    }
    Vector<unsigned int,Dim> cell_num = (this->grid_)->cellNum();
    Vector<unsigned int,Dim> &index = this->index_; //for ease of coding
    for(unsigned int i = Dim-1; i >= 0; --i)
    {
        if((index[i]==0) && (i!=0))
            index[i] = cell_num[i]-1;
        else
        {
            --index[i];
            break;
        }
    }
    return *this;
}

template <typename Scalar,int Dim>
GridCellIterator<Scalar,Dim> GridCellIterator<Scalar,Dim>::operator++ (int)
{
    if(this->grid_==NULL)
    {
        std::cerr<<"Undefined operator ++ for uninitialized iterator!\n";
        std::exit(EXIT_FAILURE);
    }
    GridCellIterator<Scalar,Dim> iterator(*this);
    Vector<unsigned int,Dim> cell_num = (this->grid_)->cellNum();
    Vector<unsigned int,Dim> &index = this->index_; //for ease of coding
    for(unsigned int i = Dim-1; i >= 0; --i)
    {
        if((index[i]==cell_num[i]-1) && (i!=0))
            index[i] = 0;
        else
        {
            ++index[i];
            break;
        }
    }
    return iterator;
}

template <typename Scalar,int Dim>
GridCellIterator<Scalar,Dim> GridCellIterator<Scalar,Dim>::operator-- (int)
{
    if(this->grid_==NULL)
    {
        std::cerr<<"Undefined operator -- for uninitialized iterator!\n";
        std::exit(EXIT_FAILURE);
    }
    GridCellIterator<Scalar,Dim> iterator(*this);
    Vector<unsigned int,Dim> cell_num = (this->grid_)->cellNum();
    Vector<unsigned int,Dim> &index = this->index_; //for ease of coding
    for(unsigned int i = Dim-1; i >= 0; --i)
    {
        if((index[i]==0) && (i!=0))
            index[i] = cell_num[i]-1;
        else
        {
            --index[i];
            break;
        }
    }
    return iterator;
}

template <typename Scalar,int Dim>
GridCellIterator<Scalar,Dim> GridCellIterator<Scalar,Dim>::operator+ (int stride) const
{
    if(this->grid_==NULL)
    {
        std::cerr<<"Undefined operator + for uninitialized iterator!\n";
        std::exit(EXIT_FAILURE);
    }
    GridCellIterator<Scalar,Dim> iterator(*this);
    Vector<unsigned int,Dim> cell_num = (iterator.grid_)->cellNum();
    Vector<unsigned int,Dim> &index = iterator.index_;
    unsigned int flat_index = GridIteratorBase<Scalar,Dim>::flatIndex(index,cell_num);
    flat_index += stride;
    index = GridIteratorBase<Scalar,Dim>::indexFromFlat(flat_index,cell_num);
    return iterator;
}

template <typename Scalar,int Dim>
GridCellIterator<Scalar,Dim> GridCellIterator<Scalar,Dim>::operator- (int stride) const
{
    if(this->grid_==NULL)
    {
        std::cerr<<"Undefined operator - for uninitialized iterator!\n";
        std::exit(EXIT_FAILURE);
    }
    GridCellIterator<Scalar,Dim> iterator(*this);
    Vector<unsigned int,Dim> cell_num = (iterator.grid_)->cellNum();
    Vector<unsigned int,Dim> &index = iterator.index_;
    unsigned int flat_index = GridIteratorBase<Scalar,Dim>::flatIndex(index,cell_num);
    flat_index += stride;
    index = GridIteratorBase<Scalar,Dim>::indexFromFlat(flat_index,cell_num);
    return iterator;
}

template <typename Scalar,int Dim>
Vector<unsigned int,Dim> GridCellIterator<Scalar,Dim>::cellIndex() const
{
    if(this->validCheck()==false)
    {
        std::cerr<<"cellIndex(): Invalid iterator!\n";
        std::exit(EXIT_FAILURE);
    }
    return this->index_;
}

template <typename Scalar,int Dim>
bool GridCellIterator<Scalar,Dim>::validCheck() const
{
    if(this->grid_==NULL)
        return false;
    Vector<unsigned int,Dim> cell_num = (this->grid_)->cellNum();
    if(GridIteratorBase<Scalar,Dim>::indexCheck(cell_num)==false)
        return false;
    return true;
}

//explicit instantiation
template class GridIteratorBase<float,2>;
template class GridIteratorBase<float,3>;
template class GridIteratorBase<double,2>;
template class GridIteratorBase<double,3>;
template class GridNodeIterator<float,2>;
template class GridNodeIterator<float,3>;
template class GridNodeIterator<double,2>;
template class GridNodeIterator<double,3>;
template class GridCellIterator<float,2>;
template class GridCellIterator<float,3>;
template class GridCellIterator<double,2>;
template class GridCellIterator<double,3>;

}  //end of namespace Physika
