/*
 * @file CPDI2_update_method.cpp 
 * @Brief the particle domain update procedure introduced in paper:
 *        "Second-order convected particle domain interpolation with enrichment for weak
 *         discontinuities at material interfaces"
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

#include "Physika_Dynamics/MPM/CPDI_mpm_solid.h"
#include "Physika_Dynamics/MPM/CPDI_Update_Methods/CPDI2_update_method.h"

namespace Physika{

template <typename Scalar, int Dim>
CPDI2UpdateMethod<Scalar,Dim>::CPDI2UpdateMethod()
    :cpdi_driver_(NULL)
{
}

template <typename Scalar, int Dim>
CPDI2UpdateMethod<Scalar,Dim>::~CPDI2UpdateMethod()
{
}

template <typename Scalar, int Dim>
void CPDI2UpdateMethod<Scalar,Dim>::updateParticleDomain()
{
//TO DO
}

//explicit instantiations
template class CPDI2UpdateMethod<float,2>;
template class CPDI2UpdateMethod<double,2>;
template class CPDI2UpdateMethod<float,3>;
template class CPDI2UpdateMethod<double,3>;

}  //end of namespace Physika
