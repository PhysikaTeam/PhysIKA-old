/*
 * @file mpm_step_method.h 
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

#ifndef PHYSIKA_DYNAMICS_MPM_MPM_STEP_METHOD_H_
#define PHYSIKA_DYNAMICS_MPM_MPM_STEP_METHOD_H_

namespace Physika{

template <typename Scalar, int Dim> class MPMBase;

template <typename Scalar, int Dim>
class MPMStepMethod
{
public:
    MPMStepMethod();
    virtual ~MPMStepMethod();
    virtual void advanceStep(Scalar dt) = 0;
    void setMPMDriver(MPMBase<Scalar,Dim> *mpm_driver);
protected:
    MPMBase<Scalar,Dim> *mpm_driver_;
}; 

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_MPM_STEP_METHOD_H_
