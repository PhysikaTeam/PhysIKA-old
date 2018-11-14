/*
 * @file PDM_bond.cpp 
 * @Basic PDMBond class. bond based of PDM
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

template<typename Scalar, int Dim>
PDMBond<Scalar, Dim>::PDMBond()
	:PDMBase(),is_homogeneous_bulk_modulus_(true)
{
	this->bulk_modulus_vec_.push_back(0);
}

template<typename Scalar, int Dim>
PDMBond<Scalar, Dim>::PDMBond(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
	:PDMBase(start_frame, end_frame, frame_rate, max_dt, write_to_file),is_homogeneous_bulk_modulus_(true)
{
	this->bulk_modulus_vec_.push_back(0);
}

template<typename Scalar, int Dim>
PDMBond<Scalar, Dim>::PDMBond(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file, Scalar bulk_modulus)
	:PDMBase(start_frame,end_frame,frame_rate,max_dt,write_to_file),is_homogeneous_bulk_modulus_(true)
{
	this->bulk_modulus_vec_.push_back(bulk_modulus);
}

template<typename Scalar, int Dim>
PDMBond<Scalar, Dim>::~PDMBond()
{

}

template<typename Scalar, int Dim>
bool PDMBond<Scalar, Dim>::isHomogeneousBulkModulus()const
{
	return this->is_homogeneous_bulk_modulus_;
}

template<typename Scalar, int Dim>
Scalar PDMBond<Scalar, Dim>::bulkModulus(unsigned int par_idx) const
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

template<typename Scalar, int Dim>
void PDMBond<Scalar, Dim>::setBulkModulus(unsigned int par_idx, Scalar bulk_modulus)
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

template<typename Scalar, int Dim>
void PDMBond<Scalar, Dim>::setHomogeneousBulkModulus(Scalar bulk_modulus)
{
	this->is_homogeneous_bulk_modulus_ = true;
	this->bulk_modulus_vec_.clear();
	this->bulk_modulus_vec_.push_back(bulk_modulus);
}

template<typename Scalar, int Dim>
void PDMBond<Scalar, Dim>::setBulkModulusVec(const std::vector<Scalar>& bulk_modulus_vec)
{
	if(bulk_modulus_vec.size()<this->particles_.size())
	{
		std::cerr<<"the size of BulkModulus_vec must be no less than the number of particles.\n ";
		std::exit(EXIT_FAILURE);
	}
	this->is_homogeneous_bulk_modulus_ = false;
	this->bulk_modulus_vec_ = bulk_modulus_vec;
}

// explicit instantiations
template class PDMBond<float,3>;
template class PDMBond<double,3>;

}// end of namespace Physika