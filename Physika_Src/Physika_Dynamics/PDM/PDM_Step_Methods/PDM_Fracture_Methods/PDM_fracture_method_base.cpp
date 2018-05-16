/*
 * @file PDM_fraceture_method_base.cpp 
 * @brief base class of fracture method for PDM drivers.
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

#include <fstream>

#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Fracture_Methods/PDM_fracture_method_base.h"
#include "Physika_Dynamics/PDM/PDM_base.h"
#include "Physika_Dynamics/PDM/PDM_particle.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMFractureMethodBase<Scalar,Dim>::PDMFractureMethodBase()
    :driver_(NULL),alpha_(0.0),enhanced_s_times_(0.0)
{
   
}

template <typename Scalar, int Dim>
PDMFractureMethodBase<Scalar,Dim>::PDMFractureMethodBase(Scalar critical_s)
    :driver_(NULL),alpha_(0.0),enhanced_s_times_(0.0)
{

}

template <typename Scalar, int Dim>
PDMFractureMethodBase<Scalar,Dim>::~PDMFractureMethodBase()
{

}


template <typename Scalar, int Dim>
void PDMFractureMethodBase<Scalar,Dim>::setHomogeneousCriticalStretch(Scalar critical_s)
{
    if (this->driver_ == NULL)
    {
        std::cout<<"error: driver is NULL!\n";
        std::exit(EXIT_FAILURE);
    }
    for (unsigned int i=0; i<this->critical_s_vec_.size(); i++)
    {
        critical_s_vec_[i] = critical_s;
    }
}

template <typename Scalar, int Dim>
void PDMFractureMethodBase<Scalar,Dim>::setCriticalStretch(unsigned int par_idx, Scalar critical_s)
{
    if (driver_ == NULL)
    {
        std::cerr<<"error: driver is note specified!\n";
        std::exit(EXIT_FAILURE);
    }
    if (par_idx >= driver_->numSimParticles())
    {
        std::cerr<<"particle index out of range.\n";
        std::exit(EXIT_FAILURE);
    }

    this->critical_s_vec_[par_idx] = critical_s;
}

template <typename Scalar, int Dim>
void PDMFractureMethodBase<Scalar,Dim>::setCriticalStretchVec(const std::vector<Scalar> & critical_s_vec)
{
    if (driver_ == NULL)
    {
        std::cerr<<"error: driver is note specified!\n";
        std::exit(EXIT_FAILURE);
    }

    if (critical_s_vec.size() < driver_->numSimParticles())
    {
        std::cerr<<"the size of critical_s_vec must be no less than the number of particle.\n";
        std::exit(EXIT_FAILURE);
    }
    this->critical_s_vec_ = critical_s_vec;
}

template <typename Scalar, int Dim>
Scalar PDMFractureMethodBase<Scalar,Dim>::criticalStretch(unsigned int par_idx) const
{

    if (driver_ == NULL)
    {
        std::cerr<<"error: driver is note specified!\n";
        std::exit(EXIT_FAILURE);
    }

    if (par_idx >= driver_->numSimParticles())
    {
        std::cerr<<"particle index out of range.\n";
        std::exit(EXIT_FAILURE);
    }
    return this->critical_s_vec_[par_idx];

}

template <typename Scalar, int Dim>
void PDMFractureMethodBase<Scalar,Dim>::setDriver(PDMBase<Scalar,Dim> * driver)
{
    if (driver == NULL)
    {
        std::cerr<<"error: can't set NULL driver to Fracture Method!\n";
        std::exit(EXIT_FAILURE);
    }
    this->driver_ = driver;
    this->critical_s_vec_.resize(this->driver_->numSimParticles());
    for (unsigned int i=0; i<this->critical_s_vec_.size(); i++)
    {
        this->critical_s_vec_[i] = 0.2; //default: 0.2
    }
}

template <typename Scalar, int Dim>
void PDMFractureMethodBase<Scalar,Dim>::setAlpha(Scalar alpha)
{
    this->alpha_ = alpha;
}

template <typename Scalar, int Dim>
void PDMFractureMethodBase<Scalar,Dim>::setEnhancedStretchTimes(Scalar enhanced_s_times)
{
    this->enhanced_s_times_ = enhanced_s_times;
}

template <typename Scalar, int Dim>
Scalar PDMFractureMethodBase<Scalar,Dim>::alpha() const
{
    return this->alpha_;
}

template <typename Scalar, int Dim>
bool PDMFractureMethodBase<Scalar,Dim>::applyFracture(Scalar s, unsigned int par_idx, std::list<PDMFamily<Scalar, Dim> > & family, typename std::list<PDMFamily<Scalar, Dim> >::iterator test_par_iter)
{
    /*
    PDMParticle<Scalar, Dim> & particle = this->driver_->particle(par_idx);
    unsigned int deleted_num = particle.initFamilySize() - particle.validFamilySize();
    unsigned int init_num = particle.initFamilySize();

    Scalar delta = this->driver_->delta(par_idx);
    Scalar rest_rel_norm = test_par_iter->restRelativePosNorm();
    Scalar w = delta/rest_rel_norm;

    Scalar k1 = 0.0;
    Scalar k2 = 1.0;

    Scalar crit_s = this->criticalStretch(par_idx)-this->alpha_*min(static_cast<Scalar>(s), static_cast<Scalar>(0.0)); 
    //crit_s *= (1.0 + k1*static_cast<Scalar>(deleted_num)/init_num)*(1.0 - k2*(rest_rel_norm - this->driver_->minDistance())/delta);
    crit_s *= (1.0 + k1*static_cast<Scalar>(deleted_num)/init_num)*w;

    if (abs(s) > crit_s)
    {
        particle.deleteFamily(test_par_iter);
        return true;
    }
    else
    {
        return false;
    }
    */
    
    PDMParticle<Scalar, Dim> & particle = this->driver_->particle(par_idx);
    unsigned int test_par_iter_id = (*test_par_iter).id();

    Scalar fracture_ratio = 1.0 - particle.validFamilySize()/static_cast<Scalar>(particle.initFamilySize());
    Scalar practical_stretch = (1.0 + this->enhanced_s_times_*fracture_ratio)*this->criticalStretch(test_par_iter_id) - alpha_*min(static_cast<Scalar>(s), static_cast<Scalar>(0.0));

    if (abs(s) > practical_stretch )
    {
        particle.deleteFamily(test_par_iter);
        return true;
    }
    else
    {
        return false;
    }
    
    
}

// explicit instantiations
template class PDMFractureMethodBase<float,2>;
template class PDMFractureMethodBase<float,3>;
template class PDMFractureMethodBase<double,2>;
template class PDMFractureMethodBase<double,3>;

}