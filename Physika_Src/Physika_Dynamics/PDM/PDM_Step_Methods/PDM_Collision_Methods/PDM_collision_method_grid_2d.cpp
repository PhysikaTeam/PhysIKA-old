/*
 * @file PDM_collision_method_grid_2d.cpp 
 * @brief base class of collision method(two dim) for PDM drivers.
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Dynamics/PDM/PDM_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_grid_2d.h"

namespace Physika{

template<typename Scalar>
PDMCollisionMethodGrid<Scalar,2>::PDMCollisionMethodGrid()
    :lambda_(0),x_spacing_(0),y_spacing_(0),
    x_bin_num_(0),y_bin_num_(0),
    bin_start_point_(0),space_hash_bin_(0)
{

}

template<typename Scalar>
PDMCollisionMethodGrid<Scalar,2>::~PDMCollisionMethodGrid()
{
    if (space_hash_bin_)
    {
        delete [] space_hash_bin_;
    }
}

template <typename Scalar>
Scalar PDMCollisionMethodGrid<Scalar,2>::lambda() const
{
    return this->lambda_;
}

template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,2>::setLambda(Scalar lambda)
{
    this->lambda_ = lambda;
}


template <typename Scalar>
Vector<Scalar,2> PDMCollisionMethodGrid<Scalar,2>::binStartPoint() const
{
    return this->bin_start_point_;
}

template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,2>::setBinStartPoint(Vector<Scalar,2> bin_start_point)
{
    this->bin_start_point_ = bin_start_point;
}

template <typename Scalar>
Scalar PDMCollisionMethodGrid<Scalar,2>::XSpacing() const
{
    return this->x_spacing_;
}

template <typename Scalar>
Scalar PDMCollisionMethodGrid<Scalar,2>::YSpacing() const
{
    return this->y_spacing_;
}


template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,2>::setXYSpacing(Scalar x_spacing, Scalar y_spacing)
{
    PHYSIKA_ASSERT(x_spacing>0.0 && y_spacing>0.0);
    this->x_spacing_ = x_spacing;
    this->y_spacing_ = y_spacing;
}

template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,2>::setUnifySpacing(Scalar spacing)
{
    PHYSIKA_ASSERT(spacing>0.0);
    this->x_spacing_ = spacing;
    this->y_spacing_ = spacing;
}

template <typename Scalar>
unsigned int PDMCollisionMethodGrid<Scalar,2>::XBinNum() const
{
    return this->x_bin_num_;
}

template <typename Scalar>
unsigned int PDMCollisionMethodGrid<Scalar,2>::YBinNum() const
{
    return this->y_bin_num_;
}


template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,2>::setXYBinNum(unsigned int x_bin_num, unsigned int y_bin_num)
{
    this->x_bin_num_ = x_bin_num;
    this->y_bin_num_ = y_bin_num;
    if (space_hash_bin_)
    {
        delete [] space_hash_bin_;
    }
    this->space_hash_bin_ = new std::vector<unsigned int>(x_bin_num_*y_bin_num_);
}

template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,2>::setUnifyBinNum(unsigned int bin_num)
{
    this->x_bin_num_ = bin_num;
    this->y_bin_num_ = bin_num;
    if (space_hash_bin_)
    {
        delete [] space_hash_bin_;
    }
    this->space_hash_bin_ = new std::vector<unsigned int>(x_bin_num_*y_bin_num_);
}

template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,2>::resetHashBin()
{
    unsigned int num_bins = x_bin_num_*y_bin_num_;
    for (unsigned int bin_idx =0; bin_idx<num_bins; bin_idx++)
    {
        space_hash_bin_[bin_idx].clear();
    }
}

template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,2>::locateParticleBin()
{
    PHYSIKA_ASSERT(driver_);
    unsigned int num_particles = driver_->numSimParticles();
    for (unsigned int par_idx =0; par_idx<num_particles; par_idx++)
    {
        Vector<Scalar, 2> par_pos = driver_->particleCurrentPosition(par_idx);
        par_pos -= bin_start_point_;
        long bin_x_pos = par_pos[0]/x_spacing_;
        long bin_y_pos = par_pos[1]/y_spacing_;

        if(isOutOfRange(bin_x_pos, bin_y_pos) == false)
        {
            unsigned int bin_pos = getHashBinPos(x_bin_num_, y_bin_num_);
            space_hash_bin_[bin_pos].push_back(par_idx);
        }
    }
}

template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,2>::collisionDectectionAndResponse()
{
    // to do
}

template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,2>::collisionMethod()
{
    resetHashBin();
    locateParticleBin();
    collisionDectectionAndResponse();
}

template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,2>::initParticleFamily(Scalar max_delta)
{
    // to do
}

template <typename Scalar>
unsigned int PDMCollisionMethodGrid<Scalar,2>::getHashBinPos(unsigned int x_pos, unsigned int y_pos)
{
    // to do
    return 0;
}

template <typename Scalar>
bool PDMCollisionMethodGrid<Scalar,2>::isOutOfRange(long x_pos, long y_pos)
{
    if ( x_pos >= x_bin_num_ || x_pos <0 || y_pos >= y_bin_num_ || y_pos <0)
    {
        return true;
    }
    return false;
}

//explicit instantiations
template class PDMCollisionMethodGrid<float,2>;
template class PDMCollisionMethodGrid<double,2>;

}// end of namespace Physika