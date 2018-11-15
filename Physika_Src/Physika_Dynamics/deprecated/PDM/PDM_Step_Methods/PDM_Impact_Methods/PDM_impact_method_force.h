/*
 * @file PDM_impact_method_Force.h 
 * @brief  class of impact method based on projectile force for PDM drivers.
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

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_IMPACT_METHODS_PDM_IMPACT_METHOD_FORCE_H
#define PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_IMPACT_METHODS_PDM_IMPACT_METHOD_FORCE_H

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Impact_Methods/PDM_impact_method_base.h"
namespace Physika{

template <typename Scalar, int Dim>
class PDMImpactMethodForce: public PDMImpactMethodBase<Scalar, Dim>
{
public:
    PDMImpactMethodForce();
    ~PDMImpactMethodForce();

    //setter
    void setKs(Scalar Ks);

    virtual void applyImpact(Scalar dt);

protected:
    Scalar Ks_;

};

} // end of namespace Physika

#endif // PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_IMPACT_METHODS_PDM_IMPACT_METHOD_FORCE_H