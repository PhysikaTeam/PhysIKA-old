/*
 * @file PDM_bond_2d.cpp 
 * @Basic PDMBond class(two dimension). bond based of PDM
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

#include <numeric>
#include <fstream>
#include "Physika_Dynamics/PDM/PDM_bond.h"
#include "Physika_Dynamics/PDM/PDM_bond_2d.h"
#include "Physika_Dynamics/PDM/PDM_Plugins/PDM_plugin_base.h"

namespace Physika{

template<typename Scalar>
PDMBond<Scalar, 2>::PDMBond()
    :PDMBase(), is_homogeneous_bulk_modulus_(true), thickness_(1.0)
{
    this->bulk_modulus_vec_.push_back(0);
}

template<typename Scalar>
PDMBond<Scalar, 2>::PDMBond(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :PDMBase(start_frame, end_frame, frame_rate, max_dt, write_to_file),is_homogeneous_bulk_modulus_(true),thickness_(1.0)
{
    this->bulk_modulus_vec_.push_back(0);
}

template<typename Scalar>
PDMBond<Scalar, 2>::PDMBond(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file, Scalar bulk_modulus, Scalar thickness)
    :PDMBase(start_frame,end_frame,frame_rate,max_dt,write_to_file),is_homogeneous_bulk_modulus_(true),thickness_(thickness)
{
    this->bulk_modulus_vec_.push_back(bulk_modulus);
}

template<typename Scalar>
PDMBond<Scalar, 2>::~PDMBond()
{

}

// the following needs further consideration
template<typename Scalar>
bool PDMBond<Scalar, 2>::isHomogeneousBulkModulus()const
{
    return this->is_homogeneous_bulk_modulus_;
}

template<typename Scalar>
Scalar PDMBond<Scalar, 2>::bulkModulus(unsigned int par_idx) const
{

    if(this->is_homogeneous_bulk_modulus_)
    {
        return this->bulk_modulus_vec_[0];
    }
    else
    {	
        if(par_idx >= this->bulk_modulus_vec_.size())
        {
            std::cerr<<"Particle index out of range.\n";
            std::exit(EXIT_FAILURE);
        }
        return this->bulk_modulus_vec_[par_idx];
    }
}

template<typename Scalar>
void PDMBond<Scalar, 2>::setBulkModulus(unsigned int par_idx, Scalar bulk_modulus)
{
    if(this->is_homogeneous_bulk_modulus_ == true)
    {
        std::cerr<<"the BulkModulus of material is homogeneous.\n";
        std::exit(EXIT_FAILURE);
    }
    if(par_idx >= this->particles_.size())
    {
        std::cerr<<"Particle index out of range.\n";
        std::exit(EXIT_FAILURE);
    }
    this->bulk_modulus_vec_[par_idx] = bulk_modulus;
}

template<typename Scalar>
void PDMBond<Scalar, 2>::setHomogeneousBulkModulus(Scalar bulk_modulus)
{
    this->is_homogeneous_bulk_modulus_ = true;
    this->bulk_modulus_vec_.clear();
    this->bulk_modulus_vec_.push_back(bulk_modulus);
}

template<typename Scalar>
void PDMBond<Scalar, 2>::setBulkModulusVec(const std::vector<Scalar>& bulk_modulus_vec)
{
    if(bulk_modulus_vec.size()<this->particles_.size())
    {
        std::cerr<<"the size of BulkModulus_vec must be no less than the number of particles.\n ";
        std::exit(EXIT_FAILURE);
    }
    this->is_homogeneous_bulk_modulus_ = false;
    this->bulk_modulus_vec_ = bulk_modulus_vec;
}

template<typename Scalar>
Scalar PDMBond<Scalar, 2>::thickness()const
{
    return this->thickness_;
}

template<typename Scalar>
void PDMBond<Scalar,2>::setThickness(Scalar thickness)
{
    this->thickness_ = thickness;
}

template class PDMBond<float,2>;
template class PDMBond<double,2>;

}// end of namespace Physika