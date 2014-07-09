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
#include "Physika_Core/Arrays/array_manager.h"
#include "Physika_Geometry/Cartesian_Grids/grid.h"
#include "Physika_Geometry/Cartesian_Grids/grid_iterator.h"
#include "Physika_Core/Range/range.h"


namespace Physika{

const int SPH_NEIGHBOR_SIZE = 150;
const int SPH_NEIGHBOR_SEGMENT = 20;

template<typename Scalar>
class NeighborList
{
public:

    NeighborList() {size_ = 0; }
    ~NeighborList(){};
public:
    unsigned int size_;
    unsigned int ids_[SPH_NEIGHBOR_SIZE];
    Scalar distance_[SPH_NEIGHBOR_SEGMENT];
};

template<typename Scalar, int Dim>
class GridQuery 
{
public:
    GridQuery();
    GridQuery(const Scalar& in_spacing, const Range<Scalar, Dim>& range_limit );
    ~GridQuery();

    void getNeighbors(const Vector<Scalar, Dim>& in_position,const Scalar& in_radius, NeighborList<Scalar>& out_neighbor_list);
    void getSizedNeighbors(const Vector<Scalar, Dim>& in_position,const Scalar& in_radius, NeighborList<Scalar>& out_neighbor_list, int in_max_num);

    void construct(const Array<Vector<Scalar, Dim>>& in_positions, ArrayManager& sim_data);
   // void construct();

private:
    void computeBoundingBox();
    unsigned int computeGridSize();
    void expandBoundingBox(const Scalar& in_padding);
    void allocMemory();

    inline unsigned int getId(Vector<Scalar, Dim> pos); 
    inline unsigned int getId(unsigned int x, unsigned int y);
    inline unsigned int getId(unsigned int x, unsigned int y, unsigned int z);

protected:
   
    unsigned int grid_num_;
    unsigned int x_num_, y_num_, z_num_;
    Array<unsigned int> begin_lists_;
    Array<unsigned int> end_lists_;

    Range<Scalar, Dim> range_;
    Range<Scalar, Dim> range_limit_;

    unsigned int particles_num_;
    Array<Vector<Scalar, Dim>> ref_positions_;
    Scalar space_;

};




}//end of namespace Physika

#endif //PHYSIKA_DYNAMICS_SPH_SPH_NEIGHBOR_QUERY_H_