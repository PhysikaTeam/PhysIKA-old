/*
 * @file PDM_collision_method_base.cpp 
 * @brief base class of collision method for PDM drivers.
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
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_base.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMCollisionMethodBase<Scalar, Dim>::PDMCollisionMethodBase()
    :driver_(NULL)
{

}

template <typename Scalar, int Dim>
PDMCollisionMethodBase<Scalar, Dim>::~PDMCollisionMethodBase()
{

}

template <typename Scalar, int Dim>
void PDMCollisionMethodBase<Scalar, Dim>::setDriver(PDMBase<Scalar, Dim> * driver)
{
    if (driver == NULL)
    {
        std::cerr<<"error: can't specify NULL driver to collision method!\n";
        std::exit(EXIT_FAILURE);
    }
    this->driver_ = driver;
}

template <typename Scalar, int Dim>
Scalar PDMCollisionMethodBase<Scalar,Dim>::Kc() const
{
    return this->Kc_;
}

template <typename Scalar, int Dim>
void PDMCollisionMethodBase<Scalar,Dim>::setKc(Scalar Kc)
{
    this->Kc_ = Kc;
}

//explicit instantiations
template class PDMCollisionMethodBase<float,2>;
template class PDMCollisionMethodBase<double,2>;
template class PDMCollisionMethodBase<float,3>;
template class PDMCollisionMethodBase<double,3>;

}// end of namespace Physika