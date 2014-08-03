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

#include "Physika_Geometry/Cartesian_Grids/grid.h"
#include "Physika_Dynamics/MPM/Weight_Function_Influence_Iterators/uniform_grid_weight_function_influence_iterator.h"

namespace Physika{

template <typename Scalar, int Dim>
UniformGridWeightFunctionInfluenceIterator<Scalar,Dim>::
UniformGridWeightFunctionInfluenceIterator(const Grid<Scalar,Dim> &grid, const Vector<Scalar,Dim> &influence_center,
                                           const Vector<Scalar,Dim> &influence_domain_scale)
 :grid_(&grid),influence_center_(influence_center),influence_domain_scale_(influence_domain_scale)
{
//TO DO
}

template <typename Scalar, int Dim>
UniformGridWeightFunctionInfluenceIterator<Scalar,Dim>::
UniformGridWeightFunctionInfluenceIterator(const UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> &iterator)
    :grid_(iterator.grid_),influence_center_(iterator.influence_center_),
     influence_domain_scale_(iterator.influence_domain_scale_),node_idx_(iterator.node_idx_)
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
    influence_center_ = iterator.influence_center_;
    influence_domain_scale_ = iterator.influence_domain_scale_;
    node_idx_ = iterator.node_idx_;
    return *this;
}

template <typename Scalar, int Dim>
bool UniformGridWeightFunctionInfluenceIterator<Scalar,Dim>::valid() const
{
//TO DO
    return false;
}

template <typename Scalar, int Dim>
UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> UniformGridWeightFunctionInfluenceIterator<Scalar,Dim>::next() const
{
//TO DO
    return *this;
}

template <typename Scalar, int Dim>
Vector<unsigned int,Dim> UniformGridWeightFunctionInfluenceIterator<Scalar,Dim>::nodeIndex() const
{
//TO DO
    return this->node_idx_;
}

//explicit instantiations
template class UniformGridWeightFunctionInfluenceIterator<float,2>;
template class UniformGridWeightFunctionInfluenceIterator<float,3>;
template class UniformGridWeightFunctionInfluenceIterator<double,2>;
template class UniformGridWeightFunctionInfluenceIterator<double,3>;

}  //end of namespace Physika
