/*
 * @file PDM_step_method_bond_verlet.h 
 * @Basic PDMStepMethodBondVerlet class. Velocity Verlet step method class, Velocity Verlet step method implemented
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

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_STEP_METHOD_BOND_VERLET_H
#define PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_STEP_METHOD_BOND_VERLET_H

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_step_method_bond.h"

namespace Physika{

// note: Instead of a basic Verlet method, Velocity Verlet method is implemented herein.

template <typename Scalar, int Dim>
class PDMStepMethodBondVerlet: public PDMStepMethodBond<Scalar, Dim>
{
public:
	PDMStepMethodBondVerlet();
	PDMStepMethodBondVerlet(PDMBond<Scalar, Dim> * pdm_base);
	virtual ~PDMStepMethodBondVerlet();

	virtual void advanceStep(Scalar dt);
};

} // end of namespace Physika

#endif // PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_STEP_METHOD_BOND_VERLET_H