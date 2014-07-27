/*
 * @file mpm_step_method.cpp 
 * @Brief base class of different methods to step one MPM time step.
 *        MPM methods conduct several operations in one time step (rasterize,
 *        update particle states, etc), the order of these operations matters.
 *        MPMStepMethod is the base class of different methods which conduct
 *        the MPM operations in different order.
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

#include <cstdlib>
#include <iostream>
#include "Physika_Dynamics/MPM/mpm_step_method.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMStepMethod<Scalar,Dim>::MPMStepMethod()
    :mpm_driver_(NULL)
{
}

template <typename Scalar, int Dim>
MPMStepMethod<Scalar,Dim>::~MPMStepMethod()
{
}

template <typename Scalar, int Dim>
void MPMStepMethod<Scalar,Dim>::setMPMDriver(MPMBase<Scalar,Dim> *mpm_driver)
{
    if(mpm_driver==NULL)
    {
        std::cerr<<"Error: cannot set NULL driver pointer to MPMStepMethod.\n";
        std::exit(EXIT_FAILURE);
    }
    mpm_driver_ = mpm_driver;
}

//explicit instantiations
template class MPMStepMethod<float,2>;
template class MPMStepMethod<double,2>;
template class MPMStepMethod<float,3>;
template class MPMStepMethod<double,3>;

}  //end of namespace Physika
