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

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

template <typename Scalar, int Dim> class Grid;

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
 */

template <typename Scalar, int Dim>
class UniformGridWeightFunctionInfluenceIterator
{
public:
    UniformGridWeightFunctionInfluenceIterator(const Grid<Scalar,Dim> &grid, const Vector<Scalar,Dim> &influence_center,
                                               const Vector<Scalar,Dim> &influence_domain_scale);
    UniformGridWeightFunctionInfluenceIterator(const UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> &iterator);
    ~UniformGridWeightFunctionInfluenceIterator();
    UniformGridWeightFunctionInfluenceIterator<Scalar,Dim>& operator= (const UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> &iterator);

    bool valid() const;
    UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> next() const;
    Vector<unsigned int,Dim> nodeIndex() const;
protected:
    const Grid<Scalar,Dim> *grid_;  //reference to grid
    Vector<Scalar,Dim> influence_center_; //center of the influence domain
    Vector<Scalar,Dim> influence_domain_scale_; //the scale between weight function influence domain and grid cell size
    Vector<unsigned int,Dim> node_idx_; //node index that current iterator points to
    //prohibit default constructor
    UniformGridWeightFunctionInfluenceIterator();
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_WEIGHT_FUNCTION_INFLUENCE_ITERATORS_UNIFORM_GRID_WEIGHT_FUNCTION_INFLUENCE_ITERATOR_H_
