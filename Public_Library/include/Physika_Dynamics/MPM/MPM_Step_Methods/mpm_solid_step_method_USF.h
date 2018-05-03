/*
 * @file mpm_solid_step_method_USF.h 
 * @Brief the USF (update stress first) method, the stress state of the particles are
 *        updated at the beginning of each time step.
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

#ifndef PHYSIKA_DYNAMICS_MPM_MPM_STEP_METHODS_MPM_SOLID_STEP_METHOD_USF_H_
#define PHYSIKA_DYNAMICS_MPM_MPM_STEP_METHODS_MPM_SOLID_STEP_METHOD_USF_H_

#include "Physika_Dynamics/MPM/MPM_Step_Methods/mpm_step_method.h"

namespace Physika{

template <typename Scalar, int Dim>
class MPMSolidStepMethodUSF: public MPMStepMethod<Scalar,Dim>
{
public:
    MPMSolidStepMethodUSF();
    ~MPMSolidStepMethodUSF();
    void advanceStep(Scalar dt);
protected:
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_MPM_STEP_METHODS_MPM_SOLID_STEP_METHOD_USF_H_
