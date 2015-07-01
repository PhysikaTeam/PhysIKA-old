/*
 * @file mpm_contact_method.cpp
 * @Brief base class of all mpm contact methods. The contact methods are alternatives
 *        to the default action of MPM which resolves contact automatically via the backgroud
 *        grid
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <cstdlib>
#include <iostream>
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Dynamics/MPM/mpm_base.h"
#include "Physika_Dynamics/MPM/MPM_Contact_Methods/mpm_contact_method.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMContactMethod<Scalar,Dim>::MPMContactMethod()
    :mpm_driver_(NULL)
{
}
 
template <typename Scalar, int Dim>
MPMContactMethod<Scalar,Dim>::~MPMContactMethod()
{
}
   
template <typename Scalar, int Dim>
void MPMContactMethod<Scalar,Dim>::setMPMDriver(MPMBase<Scalar,Dim> *mpm_driver)
{
    if(mpm_driver == NULL)
        throw PhysikaException("Cannot set NULL MPM driver to MPMContactMethod!");
    this->mpm_driver_ = mpm_driver;
}

//explicit instantiations
template class MPMContactMethod<float,2>;
template class MPMContactMethod<double,2>;
template class MPMContactMethod<float,3>;
template class MPMContactMethod<double,3>;
    
}  //end of namespace Physika
