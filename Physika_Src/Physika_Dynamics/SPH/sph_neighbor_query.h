/*
* @file sph_neighbor_query.h 
* @Basic sph neighbor query class, offer neighbor query data structure and interface
* @author Sheng Yang
* 
* This file is part of Physika, a versatile physics simulation library.
* Copyright (C) 2013 Physika Group.
*
* This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
* If a copy of the GPL was not distributed with this file, you can obtain one at:
* http://www.gnu.org/licenses/gpl-2.0.html
*
*/

#ifndef PHYSIKA_DYNAMICS_SPH_SPH_NEIGHBOR_QUERY_H_
#define PHYSIKA_DYNAMICS_SPH_SPH_NEIGHBOR_QUERY_H_

#include "Physika_Core/Vectors/vector.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Arrays/array.h"
#include "Physika_Geometry/Cartesian_Grids/grid.h"
#include "Physika_Geometry/Cartesian_Grids/grid_iterator.h"
#include "Physika_Core/Range/range.h"


namespace Physika{

const int NEIGHBOR_SIZE = 150;
const int NEIGHBOR_SEGMENT = 20;

template<typename Scalar>
class NeighborList
{
public:

    NeighborList() {size_ = 0; }
    ~NeighborList(){};
public:
    int size_;
    int ids_[NEIGHBOR_SIZE];
    Scalar distance_[NEIGHBOR_SEGMENT];
};

template<typename Scalar, int Dim>
class GridQuery 
{
public:
    GridQuery();
    ~GridQuery();

    void getNeighbors(Vector<Scalar, Dim>& in_pos, Scalar in_radius, NeighborList<Scalar>& out_neighborList);
    void getSizedNeighbors(Vector<Scalar, Dim>& in_pos, Scalar in_radius, NeighborList<Scalar>& out_neighborList, int in_maxN);

    void construct(Array<Vector<Scalar, Dim>>& pos, ArrayManager& simData);
    void construct();

private:
    void computeBoundingBox();
    int  computeGridSize();
    void expandBoundingBox(Scalar in_padding);
    void allocMemory();

    inline unsigned int getId(Vector<Scalar, Dim> pos); 
    inline unsigned int getId(int x, int y);
    inline unsigned int getId(int x, int y, int z);

protected:
   
    Grid<Scalar, Dim> grid_;
    unsigned int gird_num_;
    unsigned int x_num_, y_num_, z_num_;
    Array<int> begin_lists_;
    Array<int> end_lists_;

};




}//end of namespace Physika

#endif //PHYSIKA_DYNAMICS_SPH_SPH_NEIGHBOR_QUERY_H_