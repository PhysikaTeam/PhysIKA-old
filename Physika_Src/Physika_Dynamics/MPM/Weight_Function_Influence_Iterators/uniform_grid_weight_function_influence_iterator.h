/*
 * @file uniform_grid_weight_function_influence_iterator.h 
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

#ifndef PHYSIKA_DYNAMICS_MPM_WEIGHT_FUNCTION_INFLUENCE_ITERATORS_UNIFORM_GRID_WEIGHT_FUNCTION_INFLUENCE_ITERATOR_H_
#define PHYSIKA_DYNAMICS_MPM_WEIGHT_FUNCTION_INFLUENCE_ITERATORS_UNIFORM_GRID_WEIGHT_FUNCTION_INFLUENCE_ITERATOR_H_

#include "Physika_Geometry/Cartesian_Grids/grid.h"

namespace Physika{

template <typename Scalar, int Dim> class Vector;
template <typename Scalar, int Dim> class Range;

/*
 * UniformGridWeightFunctionInfluenceIterator: iterator the uniform grid nodes that is within the influence range of a weight function
 * Usage:
 * 1. define an iterator with: 
 *    (1) the grid
 *    (2) center of the influence domain
 *    (3) size scale of the influence domain relative to the grid cell size
 * 2. iterate the iterator while the iterator is valid:
 *    while(iter.valid())
 *    {
 *         Vector<unsigned int, Dim> node_idx = iter.nodeIndex();
 *         iter = iter.next();
 *    }
 *    
 *    or
 *
 *    for(UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> iter(grid,influence_center,influence_domain_scale); iter.valid(); iter = iter.next())
 *    {
 *         Vector<unsigned int, Dim> node_idx = iter.nodeIndex();
 *    }   
 *
 * Note: 
 *      next() method could be replaced with ++ operator
 */

template <typename Scalar, int Dim>
class UniformGridWeightFunctionInfluenceIterator
{
public:
    UniformGridWeightFunctionInfluenceIterator(const Grid<Scalar,Dim> &grid, const Vector<Scalar,Dim> &influence_center,
                                               const Vector<Scalar,Dim> &influence_radius_scale);
    UniformGridWeightFunctionInfluenceIterator(const UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> &iterator);
    ~UniformGridWeightFunctionInfluenceIterator();
    UniformGridWeightFunctionInfluenceIterator<Scalar,Dim>& operator= (const UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> &iterator);

    bool valid() const;
    UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> next() const;
    UniformGridWeightFunctionInfluenceIterator<Scalar,Dim>& operator++ ();
    UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> operator++ (int);
    Vector<unsigned int,Dim> nodeIndex() const;
protected:
    void initNodeIdxGrid(const Vector<Scalar,Dim> &influence_center, const Vector<Scalar,Dim> &influence_radius_scale);
protected:
    const Grid<Scalar,Dim> *grid_;  //reference to grid
    Grid<unsigned int,Dim> node_idx_grid_; //grid whose node is position is the valid grid node index
    typename Grid<unsigned int,Dim>::NodeIterator node_iter_; //current node iterator in the node index grid
    //prohibit default constructor
    UniformGridWeightFunctionInfluenceIterator();
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_WEIGHT_FUNCTION_INFLUENCE_ITERATORS_UNIFORM_GRID_WEIGHT_FUNCTION_INFLUENCE_ITERATOR_H_
