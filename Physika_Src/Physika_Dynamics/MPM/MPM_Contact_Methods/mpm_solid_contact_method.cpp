/*
 * @file mpm_solid_contact_method.cpp 
 * @Brief base class of all contact methods for mpm solid with uniform background grid.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Dynamics/MPM/MPM_Contact_Methods/mpm_solid_contact_method.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMSolidContactMethod<Scalar,Dim>::MPMSolidContactMethod()
    :MPMContactMethod<Scalar,Dim>()
{
}
    
template <typename Scalar, int Dim>
MPMSolidContactMethod<Scalar,Dim>::~MPMSolidContactMethod()
{
}
    
//explicit instantiations
template class MPMSolidContactMethod<float,2>;
template class MPMSolidContactMethod<float,3>;
template class MPMSolidContactMethod<double,2>;
template class MPMSolidContactMethod<double,3>;

}  //end of namespace Physika
