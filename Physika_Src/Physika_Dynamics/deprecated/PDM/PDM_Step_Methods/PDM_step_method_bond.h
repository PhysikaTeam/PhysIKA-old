/*
 * @file PDM_step_method_bond.h 
 * @Basic PDMStepMethodBond class(three dimension). basic step method class, a simplest and straightforward explicit step method implemented
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

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_STEP_METHOD_BOND_H
#define PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_STEP_METHOD_BOND_H

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_step_method_base.h"

namespace Physika{

template <typename Scalar, int Dim> class PDMBond;

template <typename Scalar, int Dim>
class PDMStepMethodBond:public PDMStepMethodBase<Scalar,Dim>
{
public:
    PDMStepMethodBond();
    PDMStepMethodBond(PDMBond<Scalar, Dim> * pdm_base);
    virtual ~PDMStepMethodBond();

protected:
    virtual void calculateForce(Scalar dt);
    virtual void calculateParameters(PDMBond<Scalar,2> * pdm_bond, unsigned int par_k, Scalar & c);
    virtual void calculateParameters(PDMBond<Scalar,3> * pdm_bond, unsigned int par_k, Scalar & c);
};

}// end of namespace Physika

#endif //PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_STEP_METHOD_BOND