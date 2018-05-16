/*
 * @file PDM_step_method_state.h 
 * @Basic PDMStepMethodState class(three dimension). basic step method class, a simplest and straightforward explicit step method implemented
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

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_STEP_METHOD_STATE_ELASTICITY_H
#define PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_STEP_METHOD_STATE_ELASTICITY_H

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_step_method_base.h"

namespace Physika{

template <typename Scalar, int Dim> class PDMState;

template <typename Scalar, int Dim>
class PDMStepMethodStateElasticity: public PDMStepMethodBase<Scalar,Dim>
{
public:
    PDMStepMethodStateElasticity();
    PDMStepMethodStateElasticity(PDMState<Scalar,Dim> * pdm_base);
    virtual ~PDMStepMethodStateElasticity();

protected:
    virtual void calculateForce(Scalar dt);
    Scalar calculateTheta(unsigned int par_k, Scalar d);
    void calculateParameters(PDMState<Scalar,2> * pdm_state, unsigned int par_k, Scalar & a, Scalar & b, Scalar & c, Scalar & d, DimensionTrait<2> trait);
    void calculateParameters(PDMState<Scalar,3> * pdm_state, unsigned int par_k, Scalar & a, Scalar & b, Scalar & c, Scalar & d, DimensionTrait<3> trait);

};

} // end of namespace Physika

#endif //PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_STEP_METHOD_STATE_ELASTICITY_H