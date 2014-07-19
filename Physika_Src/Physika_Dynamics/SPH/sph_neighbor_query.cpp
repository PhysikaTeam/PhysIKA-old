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

#include <cstring>
#include "Physika_Dynamics/SPH/sph_neighbor_query.h"
#include "Physika_Core/Utilities/math_utilities.h"


namespace Physika{


template<typename Scalar, int Dim>
GridQuery<Scalar, Dim>::GridQuery()
{
    grid_num_ = 0;
    particles_num_ = 0;
}
template<typename Scalar, int Dim>
GridQuery<Scalar, Dim>::~GridQuery()
{

}

template<typename Scalar, int Dim>
GridQuery<Scalar, Dim>::GridQuery(const Scalar& in_spacing, const Range<Scalar, Dim>& range_limit)
{
    grid_num_ = 0;
    particles_num_ = 0;
    range_limit_ = range_limit;
}

template<typename Scalar, int Dim>
void GridQuery<Scalar, Dim>::computeBoundingBox()
{
    Vector<Scalar, Dim> lo = range_limit_.minCorner(), hi = range_limit_.maxCorner();
    for (unsigned int i = 0; i < particles_num_; i++)
    {
        for(unsigned int j = 0; j < Dim; j++)
        {
            if(ref_positions_[i][j] < lo[j]) lo[j] = ref_positions_[i][j];
            if(ref_positions_[i][j] > hi[j]) hi[j] = ref_positions_[i][j];
        }
    }

    range_.setMinCorner(lo);
    range_.setMaxCorner(hi);

    expandBoundingBox(static_cast<Scalar>(.25)*space_);

}


template<typename Scalar, int Dim>
void GridQuery<Scalar, Dim>::expandBoundingBox(const Scalar& in_padding)
{
    //Scalar padding = in_padding;
    //Vector<Scalar, Dim> min_corner = range_.minCorner(), max_corner = range_.maxCorner();
    
    range_.setMinCorner(range_.minCorner() - in_padding);
    range_.setMaxCorner(range_.maxCorner() - in_padding);
}

template<typename Scalar, int Dim>
unsigned int GridQuery<Scalar, Dim>::computeGridSize()
{
    computeBoundingBox();
    x_num_ = static_cast<unsigned int>((range_.maxCorner()[0] - range_.minCorner()[0]) /space_ + 1);
    y_num_ = static_cast<unsigned int>((range_.maxCorner()[1] - range_.minCorner()[1]) /space_ + 1);
    if(Dim > 2)
        z_num_ = static_cast<unsigned int>((range_.maxCorner()[2] - range_.minCorner()[2]) /space_ + 1);
    if(Dim == 2)
        return x_num_*y_num_;
    else
        return x_num_*y_num_*z_num_;
}


template<typename Scalar, int Dim>
unsigned int GridQuery<Scalar, Dim>::getId(Vector<Scalar, Dim> pos)
{
    unsigned int ix,iy,iz;
    ix = static_cast<unsigned int>((pos[0] - range_.minCorner()[0])/space_);
    iy = static_cast<unsigned int>((pos[1] - range_.minCorner()[1])/space_);
    if(Dim == 2)
        return getId(ix, iy);
    iz = static_cast<unsigned int>((pos[2] - range_.minCorner()[2])/space_);
    return getId(ix, iy, iz);
}

template<typename Scalar, int Dim>
unsigned int GridQuery<Scalar, Dim>::getId(unsigned int x, unsigned int y)
{
    return x + y*x_num_;
}
template<typename Scalar, int Dim>
unsigned int GridQuery<Scalar, Dim>::getId(unsigned int x, unsigned int y, unsigned int z)
{
    return x + y*x_num_ + z*x_num_*z_num_;
}


void accumulation(unsigned int* in_arr, unsigned in_size)
{
    if(in_size <= 1)
    {
        std::cout << "Accumulation failed !!!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    for (unsigned int i = 1; i < in_size; i++)
    {
        in_arr[i] += in_arr[i-1];
    }
}
void radixSort(unsigned int* in_ids, unsigned int* out_rids, unsigned int in_size, unsigned int* out_begin_lists, unsigned int* out_end_lists, unsigned int in_bucket)
{
    if (in_bucket <= 0)
    {
        std::cout << "radix sort failed !!!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    memset(out_begin_lists, 0, sizeof(unsigned int)*in_bucket);
    memset(out_end_lists, 0, sizeof(unsigned int)*in_bucket);

    for (unsigned int i = 0; i < in_size; i++)
    {
        if (in_ids[i] >= 0 && in_ids[i] < in_bucket)
        {
            out_end_lists[in_ids[i]]++;
        }
    }

    accumulation(out_end_lists, in_bucket);

    out_begin_lists[0] = 0;
    for (unsigned int i = 1; i < in_bucket; i++)
    {
        out_begin_lists[i] = out_end_lists[i-1];
    }

    unsigned int* index = new unsigned int[in_bucket];
    memcpy(index, out_begin_lists, sizeof(unsigned int)*in_bucket);
    int totalNum = out_end_lists[in_bucket-1];
    for (unsigned int i = 0; i < in_size; i++)
    {
        if (in_ids[i] >= 0)
        {
            out_rids[index[in_ids[i]]] = i;
            index[in_ids[i]]++;
        }
        else
        {
            out_rids[totalNum] = i;
            totalNum++;
        }
    }

    delete[] index;
}
template<typename Scalar>
void kMinimum(Scalar *in_arr, unsigned int in_size, unsigned int* out_arr_index, unsigned int in_out_size)
{
    if(in_out_size > in_size)
    {
        std::cout << "The required size is larger than the input size !!!" << std::endl;
		std::exit(EXIT_FAILURE);
    }

    bool* checked = new bool[in_size];
    memset(checked, false, in_size*sizeof(bool));
    for (unsigned int i = 0; i < in_out_size; i++)
    {
        Scalar min_value = 10000000000;
        int index = -1;
        for (unsigned int j = 0; j < in_size; j++)
        {
            if(!checked[j])
            {
                if(in_arr[j] < min_value)
                {
                    min_value = in_arr[j];
                    index = j;
                }
            }
        }
        
        checked[index] = true;
        out_arr_index[i] = index;
    }
    
    delete[] checked;
}

template<typename Scalar, int Dim>
void GridQuery<Scalar, Dim>::construct(Array<Vector<Scalar, Dim>>& in_positions, ArrayManager& sim_data)
{
    particles_num_ = in_positions.elementCount();
    ref_positions_ = in_positions;

    Array<unsigned int> ref_ids, ref_reordered_ids;
    ref_ids.resize(particles_num_);
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
        ref_ids[i] = getId(ref_positions_[i]);
    }
    
    
    radixSort(ref_ids.data(), ref_reordered_ids.data(), particles_num_, begin_lists_.data(), end_lists_.data(), grid_num_);

    sim_data.permutate(ref_reordered_ids.data(), particles_num_);

}

template<typename Scalar, int Dim>
void GridQuery<Scalar, Dim>::getNeighbors(const Vector<Scalar, Dim>& in_position,const Scalar& in_radius, NeighborList<Scalar>& out_neighbor_list)
{
    out_neighbor_list.size_ = 0;
    ///Scalar radius_square = in_radius;
    unsigned int ix_st,ix_ed,iy_st,iy_ed,iz_st,iz_ed;
    Vector<Scalar, Dim> st = (in_position - in_radius - range_.minCorner())/space_;
    Vector<Scalar, Dim> ed = (in_position + in_radius - range_.minCorner())/space_;
    ix_st = static_cast<unsigned int>(st[0]);
    iy_st = static_cast<unsigned int>(st[1]);
    if(Dim == 3)
        iz_st = static_cast<unsigned int>(st[2]);
    ix_ed = static_cast<unsigned int>(ed[0]);
    iy_ed = static_cast<unsigned int>(ed[1]);
    if(Dim == 3)
        iz_ed = static_cast<unsigned int>(ed[2]);

    for (unsigned int i = ix_st; i <= ix_ed; i++)
    {
        for (unsigned int j = iy_st; j <= iy_ed; j++)
        {
            if(Dim == 3)
            {
                for (unsigned int k = iz_st; k <= iz_ed; k++)
                {
                    unsigned int grid_index = getId(i, j, k);
                    if (grid_index >= 0)
                    {
                        for (unsigned int t = begin_lists_[grid_index]; t < end_lists_[grid_index]; t++)
                        {
                            Scalar dist_square = (ref_positions_[t] - in_position).norm();
                            if (dist_square <= in_radius && out_neighbor_list.size_ < SPH_NEIGHBOR_SIZE)
                            {
                                out_neighbor_list.ids_[out_neighbor_list.size_] = t;
                                out_neighbor_list.distance_[out_neighbor_list.size_] = dist_square;
                                out_neighbor_list.size_++;
                            }
                        }
                    }
                }
            
            }
            else
            {
                unsigned int grid_index = getId(i, j);
                if (grid_index >= 0)
                {
                    for (unsigned int t = begin_lists_[grid_index]; t < end_lists_[grid_index]; t++)
                    {
                        Scalar dist_square = (ref_positions_[t] - in_position).norm();
                        if (dist_square <= in_radius && out_neighbor_list.size_ < SPH_NEIGHBOR_SIZE)
                        {
                            out_neighbor_list.ids_[out_neighbor_list.size_] = t;
                            out_neighbor_list.distance_[out_neighbor_list.size_] = dist_square;
                            out_neighbor_list.size_++;
                        }
                    }
                }
            }
         
        }
    }

}

template<typename Scalar, int Dim>
void GridQuery<Scalar, Dim>::getSizedNeighbors(const Vector<Scalar, Dim>& in_position,const Scalar& in_radius, NeighborList<Scalar>& out_neighbor_list, unsigned int in_max_num)
{
    getNeighbors(in_position, in_radius, out_neighbor_list);
    if(out_neighbor_list.size_ > in_max_num)
    {
        unsigned int seg_size[SPH_NEIGHBOR_SEGMENT] = {0};
        unsigned int seg_ids[SPH_NEIGHBOR_SEGMENT][SPH_NEIGHBOR_SIZE];
        Scalar seg_distances[SPH_NEIGHBOR_SEGMENT][SPH_NEIGHBOR_SIZE];
        for (unsigned int i = 0; i < out_neighbor_list.size_; i++)
        {
            Scalar dist = out_neighbor_list.distance_[i];
            unsigned int index = static_cast<unsigned int >(pow(static_cast<Scalar>(0.99)*dist/in_radius, 2)*SPH_NEIGHBOR_SEGMENT);
            seg_ids[index][seg_size[index]] = out_neighbor_list.ids_[i];
            seg_distances[index][seg_size[index]] = out_neighbor_list.distance_[i];
            seg_size[index]++;
        }

        NeighborList<Scalar> sized_neighbor_list;
        unsigned int total_num = 0;
        unsigned int j;
        for (j = 0; j < SPH_NEIGHBOR_SEGMENT; j++)
        {
            total_num += seg_size[j];
            if (total_num <= in_max_num)
            {
                for (unsigned int k = 0; k < seg_size[j]; k++)
                {
                    sized_neighbor_list.ids_[sized_neighbor_list.size_] = seg_ids[j][k];
                    sized_neighbor_list.distance_[sized_neighbor_list.size_] = seg_distances[j][k];
                    sized_neighbor_list.size_++;
                }
            }
            else
                break;
        }

        unsigned int rem_num = in_max_num + seg_size[j] - total_num;
        unsigned int* rem_arr = new unsigned int[rem_num];
        kMinimum<Scalar>(seg_distances[j], seg_size[j], rem_arr, rem_num);

        for (unsigned int k = 0; k < rem_num; k++)
        {
            sized_neighbor_list.ids_[sized_neighbor_list.size_] = seg_ids[j][rem_arr[k]];
            sized_neighbor_list.distance_[sized_neighbor_list.size_] = seg_distances[j][rem_arr[k]];
            sized_neighbor_list.size_++;
        }

        out_neighbor_list = sized_neighbor_list;

        delete[] rem_arr;
    }
}


template class GridQuery<float , 3>;
template class GridQuery<double, 3>;
template class GridQuery<float, 2>;
template class GridQuery<double, 2>;

}//end of namespace Physika

