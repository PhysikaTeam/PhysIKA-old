/*
 * @file PDM_impact_method_base.cpp 
 * @brief base class of impact method for PDM drivers.
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

#include "Physika_Dynamics/PDM/PDM_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Impact_Methods/PDM_impact_method_base.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMImpactMethodBase<Scalar,Dim>::PDMImpactMethodBase()
    :driver_(NULL),impact_velocity_(0),impact_pos_(0),impact_radius_(0),trigger_special_treatment_(false)
{

}

template <typename Scalar, int Dim>
PDMImpactMethodBase<Scalar,Dim>::~PDMImpactMethodBase()
{

}

template <typename Scalar, int Dim>
void PDMImpactMethodBase<Scalar,Dim>::setDriver(PDMBase<Scalar,Dim> * driver)
{
    if (driver == NULL)
    {
        std::cerr<<"error: can't set NULL driver to Impact Method!\n";
        std::exit(EXIT_FAILURE);
    }
    this->driver_ = driver;
}

template <typename Scalar, int Dim>
void PDMImpactMethodBase<Scalar,Dim>::setImpactVelocity(const Vector<Scalar, Dim> & velocity)
{
    this->impact_velocity_ = velocity;
}

template <typename Scalar, int Dim>
void PDMImpactMethodBase<Scalar,Dim>::setImpactPos(const Vector<Scalar, Dim> & pos)
{
    this->impact_pos_ = pos;
}

template <typename Scalar, int Dim>
void PDMImpactMethodBase<Scalar,Dim>::setImpactRadius(Scalar radius)
{
    this->impact_radius_ = radius;
}

template <typename Scalar, int Dim>
const Vector<Scalar,Dim> & PDMImpactMethodBase<Scalar,Dim>::impactVelocity() const
{
    return this->impact_velocity_;
}

template <typename Scalar, int Dim>
const Vector<Scalar,Dim> & PDMImpactMethodBase<Scalar,Dim>::impactPos() const
{
    return this->impact_pos_;
}

template <typename Scalar, int Dim>
float PDMImpactMethodBase<Scalar,Dim>::impactRadius() const
{
    return this->impact_radius_;
}

template <typename Scalar, int Dim>
void PDMImpactMethodBase<Scalar,Dim>::enableTriggerSpecialTreatment()
{
    this->trigger_special_treatment_ = true;
}

template <typename Scalar, int Dim>
void PDMImpactMethodBase<Scalar,Dim>::disableTriggerSpecialTreatment()
{
    this->trigger_special_treatment_ = false;
}

template <typename Scalar, int Dim>
void PDMImpactMethodBase<Scalar,Dim>::triggerSpecialTreatment()
{
    //do nothing
}

// explicit instantiations
template class PDMImpactMethodBase<float,2>;
template class PDMImpactMethodBase<float,3>;
template class PDMImpactMethodBase<double,2>;
template class PDMImpactMethodBase<double,3>;


} // end of namespace Physika