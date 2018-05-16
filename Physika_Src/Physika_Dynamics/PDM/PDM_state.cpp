/*
 * @file PDM_state.cpp 
 * @Basic PDMState class. state based of PDM
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

#include <iostream>
#include "Physika_Dynamics/PDM/PDM_state.h"

namespace Physika{

template<typename Scalar, int Dim>
PDMState<Scalar,Dim>::PDMState()
    :PDMBond(), is_homogeneous_shear_modulus_(true)
{
    this->shear_modulus_vec_.push_back(3.0/5.0*this->bulkModulus());
}


template <typename Scalar, int Dim>
PDMState<Scalar,Dim>::PDMState(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :PDMBond(start_frame, end_frame, frame_rate, max_dt, write_to_file), is_homogeneous_shear_modulus_(true)
{
    this->shear_modulus_vec_.push_back(3.0/5.0*this->bulkModulus());
}

template <typename Scalar, int Dim>
PDMState<Scalar, Dim>::PDMState(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file, Scalar bulk_modulus, Scalar shear_modulus)
    :PDMBond(start_frame, end_frame, frame_rate, max_dt, write_to_file, bulk_modulus),is_homogeneous_shear_modulus_(true)
{
    this->shear_modulus_vec_.push_back(shear_modulus);
}


template <typename Scalar, int Dim>
PDMState<Scalar,Dim>::~PDMState()
{

}

template<typename Scalar, int Dim>
bool PDMState<Scalar, Dim>::isHomogeneousShearModulus() const
{
    return this->is_homogeneous_shear_modulus_;
}

template<typename Scalar, int Dim>
Scalar PDMState<Scalar, Dim>::shearModulus(unsigned int par_idx) const
{
    if (this->is_homogeneous_shear_modulus_)
    {
        return this->shear_modulus_vec_[0];
    }
    else
    {
        if (par_idx>=this->shear_modulus_vec_.size())
        {
            std::cerr<<"Particle index out of range.\n";
            std::exit(EXIT_FAILURE);
        }
        return this->shear_modulus_vec_[par_idx];
    }
}

template<typename Scalar, int Dim>
void PDMState<Scalar, Dim>::setShearModulus(unsigned int par_idx, Scalar mu)
{
    if (this->is_homogeneous_shear_modulus_)
    {
        std::cerr<<"the ShearModulus of material is homogeneous.\n";
        std::exit(EXIT_FAILURE);
    }
    if (par_idx >= this->particles_.size())
    {
        std::cerr<<"Particle index out of range.\n";
        std::exit(EXIT_FAILURE);
    }
    this->shear_modulus_vec_[par_idx] = mu;

}

template <typename Scalar, int Dim>
void PDMState<Scalar, Dim>::setHomogeneousShearModulus(Scalar mu)
{
    this->is_homogeneous_shear_modulus_ = true;
    this->shear_modulus_vec_.clear();
    this->shear_modulus_vec_.push_back(mu);
}

template <typename Scalar, int Dim>
void PDMState<Scalar, Dim>::setShearModulusVec(const std::vector<Scalar> mu_vec)
{
    if (mu_vec.size()<this->particles_.size())
    {
        std::cerr<<"the size of ShearModulus_vec must be no less than the number of particles.\n";
        std::exit(EXIT_FAILURE);
    }
    this->is_homogeneous_shear_modulus_ = false;
    this->shear_modulus_vec_ = mu_vec;
}

// explicit instantiations
template class PDMState<float,3>;
template class PDMState<double,3>;

}// end of namespace Physika