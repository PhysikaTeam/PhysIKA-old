/*
 * @file uniform_grid_weight_function_influence_iterator.cpp 
 * @Brief iterator of uniform grid nodes that is within influence range of a weight function.
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
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Range/range.h"
#include "Physika_Core/Grid_Weight_Functions/grid_weight_function.h"
#include "Physika_Dynamics/Utilities/Weight_Function_Influence_Iterators/uniform_grid_weight_function_influence_iterator.h"

namespace Physika{

template <typename Scalar, int Dim>
UniformGridWeightFunctionInfluenceIterator<Scalar,Dim>::
UniformGridWeightFunctionInfluenceIterator(const Grid<Scalar,Dim> &grid, const Vector<Scalar,Dim> &influence_center,
                                           const GridWeightFunction<Scalar,Dim> &weight_function)
 :grid_(&grid)
{
    initNodeIdxGrid(influence_center,weight_function.supportRadius());
    node_iter_ = node_idx_grid_.nodeBegin();
}

template <typename Scalar, int Dim>
UniformGridWeightFunctionInfluenceIterator<Scalar,Dim>::
UniformGridWeightFunctionInfluenceIterator(const UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> &iterator)
    :grid_(iterator.grid_),node_idx_grid_(iterator.node_idx_grid_),node_iter_(iterator.node_iter_)
{
}

template <typename Scalar, int Dim>
UniformGridWeightFunctionInfluenceIterator<Scalar,Dim>::
~UniformGridWeightFunctionInfluenceIterator()
{
}

template <typename Scalar, int Dim>
UniformGridWeightFunctionInfluenceIterator<Scalar,Dim>& UniformGridWeightFunctionInfluenceIterator<Scalar,Dim>::
operator= (const UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> &iterator)
{
    grid_ = iterator.grid_;
    node_idx_grid_ = iterator.node_idx_grid_;
    node_iter_ = iterator.node_iter_;
    return *this;
}

template <typename Scalar, int Dim>
bool UniformGridWeightFunctionInfluenceIterator<Scalar,Dim>::valid() const
{
    bool status = (node_iter_ != node_idx_grid_.nodeEnd());
    return status;
}

template <typename Scalar, int Dim>
UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> UniformGridWeightFunctionInfluenceIterator<Scalar,Dim>::next() const
{
    UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> result(*this);
    ++(result.node_iter_);
    return result;
}

template <typename Scalar, int Dim>
UniformGridWeightFunctionInfluenceIterator<Scalar,Dim>& UniformGridWeightFunctionInfluenceIterator<Scalar,Dim>::
operator++ ()
{
    ++node_iter_;
    return *this;
}

template <typename Scalar, int Dim>
UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> UniformGridWeightFunctionInfluenceIterator<Scalar,Dim>::
operator++ (int)
{
    UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> result(*this);
    ++node_iter_;
    return result;
}

template <typename Scalar, int Dim>
Vector<unsigned int,Dim> UniformGridWeightFunctionInfluenceIterator<Scalar,Dim>::nodeIndex() const
{
    if(this->valid()==false)
    {
        std::cerr<<"Error: invalid UniformGridWeightFunctionInfluenceIterator, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    return node_idx_grid_.node(node_iter_.nodeIndex());
}

template <typename Scalar, int Dim>
void UniformGridWeightFunctionInfluenceIterator<Scalar,Dim>::initNodeIdxGrid(const Vector<Scalar,Dim> &influence_center, Scalar influence_radius_scale)
{
    Vector<Scalar,Dim> influence_radius, cell_dx = grid_->dX();
    for(unsigned int i = 0; i < Dim; ++i)
        influence_radius[i] = cell_dx[i]*influence_radius_scale;
    Vector<Scalar,Dim> influence_domain_min_corner = influence_center - influence_radius;
    Vector<Scalar,Dim> influence_domain_max_corner = influence_center + influence_radius;
    Vector<Scalar,Dim> grid_min_corner = grid_->minCorner();
    Vector<Scalar,Dim> grid_max_corner = grid_->maxCorner();
    //clamp to grid boundary if out of range
    for(unsigned int i = 0; i < Dim; ++i)
    {
        if(influence_domain_min_corner[i]<grid_min_corner[i])
            influence_domain_min_corner[i] = grid_min_corner[i];
        if(influence_domain_max_corner[i]>grid_max_corner[i])
            influence_domain_max_corner[i] = grid_max_corner[i];
    }
    Vector<unsigned int,Dim> cell_idx;
    Vector<Scalar,Dim> weight;
    Range<unsigned int,Dim> node_idx_range;
    grid_->cellIndexAndBiasInCell(influence_domain_min_corner,cell_idx,weight);
    for(unsigned int i = 0; i < Dim; ++i)
        if(weight[i] > std::numeric_limits<Scalar>::epsilon())
            ++cell_idx[i];
    node_idx_range.setMinCorner(cell_idx);
    grid_->cellIndexAndBiasInCell(influence_domain_max_corner,cell_idx,weight);
    for(unsigned int i = 0; i < Dim; ++i)
        if(weight[i] > 1.0 - std::numeric_limits<Scalar>::epsilon())
            ++cell_idx[i];
    node_idx_range.setMaxCorner(cell_idx);
    node_idx_grid_.setDomain(node_idx_range);
    node_idx_grid_.setCellNum(node_idx_range.edgeLengths());
}

//explicit instantiations
template class UniformGridWeightFunctionInfluenceIterator<float,2>;
template class UniformGridWeightFunctionInfluenceIterator<float,3>;
template class UniformGridWeightFunctionInfluenceIterator<double,2>;
template class UniformGridWeightFunctionInfluenceIterator<double,3>;

}  //end of namespace Physika
