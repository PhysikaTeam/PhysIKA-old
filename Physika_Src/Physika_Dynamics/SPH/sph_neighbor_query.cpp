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

#include "Physika_Dynamics/SPH/sph_neighbor_query.h"

namespace Physika{

template<typename Scalar, int Dim>
GridQuery<Scalar, Dim>::GridQuery()
{

}

template<typename Scalar, int Dim>
void GridQuery<Scalar, Dim>::computeBoundingBox()
{
    //Range range;
    Vector<Scalar, Dim> lo = ref_positions_[0], hi = ref_positions_[0];
    for (unsigned int i = 0; i < particles_num_; i++)
    {
        for(unsigned int j = 0; j < Dim; j++)
        {
            if(ref_positions_[i][j] < lo[j]) lo[j] = ref_positions_[i][j];
            if(ref_positions_[i][j] > hi[j]) hi[j] = ref_positions_[i][j];
        }
    }

    lo = max(lo, range_limit_.minCorner());
    hi = min(hi, range_limit_.maxCorner());

    expandBoundingBox(0.25*space_);

}

template<typename Scalar, int Dim>
void GridQuery<Scalar, Dim>::computeGridSize()
{
    computeBoundingBox();
    x_num_ = (range_.maxCorner()[0] - range_.minCorner()[0]) /space_ + 1;
    y_num_ = (range_.maxCorner()[1] - range_.minCorner()[1]) /space_ + 1;
    if(Dim > 2)
        z_num_ = (range_.maxCorner()[2] - range_.minCorner()[2]) /space_ + 1;
    if(Dim == 2)
        return x_num_*y_num_;
    return x_num_*y_num_*z_num_;
}


template<typename Scalar, int Dim>
unsigned int GridQuery<Scalar, Dim>::getId(Vector<Scalar, Dim> pos)
{
    int ix,iy,iz;
    //return getId(ix, iy);
    return getId(ix, iy, iz);
}

template<typename Scalar, int Dim>
unsigned int GridQuery<Scalar, Dim>::getId(int x, int y, int z)
{
    return x + y*x_num_ + z*x_num_*z_num_;
}

template<typename Scalar, int Dim>
void GridQuery<Scalar, Dim>::construct(Array<Vector<Scalar, Dim>>& in_positions, ArrayManager& sim_data)
{
    particles_num_ = in_positions.elementCount();
    ref_positions_ = in_positions.data();

    Array<unsigned int> ref_ids_, ref_reordered_ids;
    ref_ids_.resize(particles_num_);
    ref_reordered_ids.resize(particles_num_);

    int grid_num = computeGridSize();
    if(grid_num != grid_num_)
    {
        if(begin_lists_.data() != NULL) begin_lists_.resize(grid_num);
        if(end_lists_.data() != NULL) end_lists_.resize(grid_num);
        grid_num_ = grid_num;
    }

    for (unsigned int i = 0; i < particles_num_; i++)
    {
        ref_ids_[i] = getId(ref_positions_[i]);
    }
    
    //radixsort to get new id reodered by positions;
    //To do reorder.
    
    

    sim_data.permutate(ref_reordered_ids.data(), particles_num_);

}

}//end of namespace Physika

