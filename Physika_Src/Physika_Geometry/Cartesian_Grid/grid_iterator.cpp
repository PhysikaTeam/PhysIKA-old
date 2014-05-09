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
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Geometry/Cartesian_Grid/grid_iterator.h"
#include "Physika_Geometry/Cartesian_Grid/grid.h"

namespace Physika{

template <typename Scalar,int Dim>
GridIteratorBase<Scalar,Dim>::GridIteratorBase()
    :index_(-1),grid_(NULL)
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
    return (index_==iterator.index_)&&(grid_==iterator.grid_);
}

template <typename Scalar,int Dim>
bool GridIteratorBase<Scalar,Dim>::operator!= (const GridIteratorBase<Scalar,Dim> &iterator) const
{
    return !(*this == iterator);
}

template <typename Scalar,int Dim>
int GridIteratorBase<Scalar,Dim>::flatIndex(const Vector<int,Dim> &index, const Vector<int,Dim> &dimension) const
{
    for(int i = Dim-1; i >=0; --i)
	if( (index[i]<0) || (index[i]>=dimension[i]) )
	    return -1;
    int flat_index = 0;
    Vector<int,Dim> vec = index;
    for(int i = 0; i < Dim; ++i)
    {
	for(int j = i+1; j < Dim; ++j)
	    vec[i] *= dimension[j];
	flat_index += vec[i];
    }
    return flat_index;
}

template <typename Scalar,int Dim>
Vector<int,Dim> GridIteratorBase<Scalar,Dim>::indexFromFlat(int flat_index, const Vector<int,Dim> &dimension) const
{
    Vector<int,Dim> index(1);
    for(int i = 0; i < Dim; ++i)
    {
	for(int j = i+1; j < Dim; ++j)
	    index[i] *= dimension[j];
	index[i] = flat_index / index[i];
	flat_index = flat_index % index[i];
	if( (index[i]<0) || (index[i] >= dimension[i]) )
	    return Vector<int,Dim>(-1);
    }
    return index;
}

template <typename Scalar,int Dim>
void GridIteratorBase<Scalar,Dim>::clampIndex(const Vector<int,Dim> &dimension)
{
    for(int i = 0; i < Dim; ++i)
    {
	if( (index_[i]<0) || (index_[i]>=dimension[i]) )
	{
	    index_ = Vector<int,Dim>(-1);  //set all entries to -1
	    break;
	}
    }
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
    return GridIteratorBase<Scalar,Dim>::operator!= (iterator);
}

template <typename Scalar,int Dim>
GridNodeIterator<Scalar,Dim>& GridNodeIterator<Scalar,Dim>::operator++ ()
{
    PHYSIKA_ASSERT(this->grid_);
    Vector<int,Dim> node_num = (this->grid_)->nodeNum();
    Vector<int,Dim> &index = this->index_; //for ease of coding
    for(int i = Dim-1; i >= 0; --i)
    {
	if(index[i]==node_num[i]-1)
	    index[i] = 0;
	else
	{
	    ++index[i];
	    break;
	}
    }
    GridIteratorBase<Scalar,Dim>::clampIndex(node_num);
    return *this;
}

template <typename Scalar,int Dim>
GridNodeIterator<Scalar,Dim>& GridNodeIterator<Scalar,Dim>::operator-- ()
{
    PHYSIKA_ASSERT(this->grid_);
    Vector<int,Dim> node_num = (this->grid_)->nodeNum();
    Vector<int,Dim> &index = this->index_; //for ease of coding
    for(int i = Dim-1; i >= 0; --i)
    {
	if(index[i]==0)
	    index[i] = node_num[i]-1;
	else
	{
	    --index[i];
	    break;
	}
    }
    GridIteratorBase<Scalar,Dim>::clampIndex(node_num);
    return *this;
}

template <typename Scalar,int Dim>
GridNodeIterator<Scalar,Dim> GridNodeIterator<Scalar,Dim>::operator++ (int)
{
    GridNodeIterator<Scalar,Dim> iterator(*this);
    PHYSIKA_ASSERT(this->grid_);
    Vector<int,Dim> node_num = (this->grid_)->nodeNum();
    Vector<int,Dim> &index = this->index_; //for ease of coding
    for(int i = Dim-1; i >= 0; --i)
    {
	if(index[i]==node_num[i]-1)
	    index[i] = 0;
	else
	{
	    ++index[i];
	    break;
	}
    }
    GridIteratorBase<Scalar,Dim>::clampIndex(node_num);
    return iterator;
}

template <typename Scalar,int Dim>
GridNodeIterator<Scalar,Dim> GridNodeIterator<Scalar,Dim>::operator-- (int)
{
    GridNodeIterator<Scalar,Dim> iterator(*this);
    PHYSIKA_ASSERT(this->grid_);
    Vector<int,Dim> node_num = (this->grid_)->nodeNum();
    Vector<int,Dim> &index = this->index_; //for ease of coding
    for(int i = Dim-1; i >= 0; --i)
    {
	if(index[i]==0)
	    index[i] = node_num[i]-1;
	else
	{
	    --index[i];
	    break;
	}
    }
    GridIteratorBase<Scalar,Dim>::clampIndex(node_num);
    return iterator;
}

template <typename Scalar,int Dim>
GridNodeIterator<Scalar,Dim> GridNodeIterator<Scalar,Dim>::operator+ (int stride) const
{
    GridNodeIterator<Scalar,Dim> iterator(*this);
    PHYSIKA_ASSERT(iterator.grid_);
    Vector<int,Dim> node_num = (iterator.grid_)->nodeNum();
    Vector<int,Dim> &index = iterator.index_;
    int flat_index = GridIteratorBase<Scalar,Dim>::flatIndex(index,node_num);
    flat_index += stride;
    index = GridIteratorBase<Scalar,Dim>::indexFromFlat(flat_index,node_num);
    return iterator;
}

template <typename Scalar,int Dim>
GridNodeIterator<Scalar,Dim> GridNodeIterator<Scalar,Dim>::operator- (int stride) const
{
    GridNodeIterator<Scalar,Dim> iterator(*this);
    PHYSIKA_ASSERT(iterator.grid_);
    Vector<int,Dim> node_num = (iterator.grid_)->nodeNum();
    Vector<int,Dim> &index = iterator.index_;
    int flat_index = GridIteratorBase<Scalar,Dim>::flatIndex(index,node_num);
    flat_index -= stride;
    index = GridIteratorBase<Scalar,Dim>::indexFromFlat(flat_index,node_num);
    return iterator;
}

template <typename Scalar,int Dim>
const Vector<int,Dim>& GridNodeIterator<Scalar,Dim>::nodeIndex() const
{
    return this->index_;
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
    return GridIteratorBase<Scalar,Dim>::operator!= (iterator);
}

template <typename Scalar,int Dim>
GridCellIterator<Scalar,Dim>& GridCellIterator<Scalar,Dim>::operator++ ()
{
    PHYSIKA_ASSERT(this->grid_);
    Vector<int,Dim> cell_num = (this->grid_)->cellNum();
    Vector<int,Dim> &index = this->index_; //for ease of coding
    for(int i = Dim-1; i >= 0; --i)
    {
	if(index[i]==cell_num[i]-1)
	    index[i] = 0;
	else
	{
	    ++index[i];
	    break;
	}
    }
    GridIteratorBase<Scalar,Dim>::clampIndex(cell_num);
    return *this;
}

template <typename Scalar,int Dim>
GridCellIterator<Scalar,Dim>& GridCellIterator<Scalar,Dim>::operator-- ()
{
    PHYSIKA_ASSERT(this->grid_);
    Vector<int,Dim> cell_num = (this->grid_)->cellNum();
    Vector<int,Dim> &index = this->index_; //for ease of coding
    for(int i = Dim-1; i >= 0; --i)
    {
	if(index[i]==0)
	    index[i] = cell_num[i]-1;
	else
	{
	    --index[i];
	    break;
	}
    }
    GridIteratorBase<Scalar,Dim>::clampIndex(cell_num);
    return *this;
}

template <typename Scalar,int Dim>
GridCellIterator<Scalar,Dim> GridCellIterator<Scalar,Dim>::operator++ (int)
{
    GridCellIterator<Scalar,Dim> iterator(*this);
    PHYSIKA_ASSERT(this->grid_);
    Vector<int,Dim> cell_num = (this->grid_)->cellNum();
    Vector<int,Dim> &index = this->index_; //for ease of coding
    for(int i = Dim-1; i >= 0; --i)
    {
	if(index[i]==cell_num[i]-1)
	    index[i] = 0;
	else
	{
	    ++index[i];
	    break;
	}
    }
    GridIteratorBase<Scalar,Dim>::clampIndex(cell_num);
    return iterator;
}

template <typename Scalar,int Dim>
GridCellIterator<Scalar,Dim> GridCellIterator<Scalar,Dim>::operator-- (int)
{
    GridCellIterator<Scalar,Dim> iterator(*this);
    PHYSIKA_ASSERT(this->grid_);
    Vector<int,Dim> cell_num = (this->grid_)->cellNum();
    Vector<int,Dim> &index = this->index_; //for ease of coding
    for(int i = Dim-1; i >= 0; --i)
    {
	if(index[i]==0)
	    index[i] = cell_num[i]-1;
	else
	{
	    --index[i];
	    break;
	}
    }
    GridIteratorBase<Scalar,Dim>::clampIndex(cell_num);
    return iterator;
}

template <typename Scalar,int Dim>
GridCellIterator<Scalar,Dim> GridCellIterator<Scalar,Dim>::operator+ (int stride) const
{
    GridCellIterator<Scalar,Dim> iterator(*this);
    PHYSIKA_ASSERT(iterator.grid_);
    Vector<int,Dim> cell_num = (iterator.grid_)->cellNum();
    Vector<int,Dim> &index = iterator.index_;
    int flat_index = GridIteratorBase<Scalar,Dim>::flatIndex(index,cell_num);
    flat_index += stride;
    index = GridIteratorBase<Scalar,Dim>::indexFromFlat(flat_index,cell_num);
    return iterator;
}

template <typename Scalar,int Dim>
GridCellIterator<Scalar,Dim> GridCellIterator<Scalar,Dim>::operator- (int stride) const
{
    GridCellIterator<Scalar,Dim> iterator(*this);
    PHYSIKA_ASSERT(iterator.grid_);
    Vector<int,Dim> cell_num = (iterator.grid_)->cellNum();
    Vector<int,Dim> &index = iterator.index_;
    int flat_index = GridIteratorBase<Scalar,Dim>::flatIndex(index,cell_num);
    flat_index += stride;
    index = GridIteratorBase<Scalar,Dim>::indexFromFlat(flat_index,cell_num);
    return iterator;
}

template <typename Scalar,int Dim>
const Vector<int,Dim>& GridCellIterator<Scalar,Dim>::cellIndex() const
{
    return this->index_;
}

//explicit instantiation
template class GridNodeIterator<float,2>;
template class GridNodeIterator<float,3>;
template class GridNodeIterator<double,2>;
template class GridNodeIterator<double,3>;
template class GridCellIterator<float,2>;
template class GridCellIterator<float,3>;
template class GridCellIterator<double,2>;
template class GridCellIterator<double,3>;

}  //end of namespace Physika
