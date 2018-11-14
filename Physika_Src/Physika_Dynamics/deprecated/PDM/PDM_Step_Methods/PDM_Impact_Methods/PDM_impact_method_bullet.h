/*
 * @file PDM_impact_method_bullet.h 
 * @brief  class of impact method based on bullet for PDM drivers.
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

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_IMPACT_METHODS_PDM_IMPACT_METHOD_BULLET_H
#define PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_IMPACT_METHODS_PDM_IMPACT_METHOD_BULLET_H

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Impact_Methods/PDM_impact_method_base.h"

namespace Physika{

    template <typename Scalar, int Dim>
    class PDMImpactMethodBullet: public PDMImpactMethodBase<Scalar,Dim>
    {
    public:
        PDMImpactMethodBullet();
        ~PDMImpactMethodBullet();

        virtual void applyImpact(Scalar dt);
        void setRecoveryCoefficient(Scalar recovery_coefficient);
        void setBulletCylinderLength(Scalar bullet_cylinder_length);
        void enableProjectParticlePos();
        void disableProjectParticlePos();

    protected:

        //Note: the bullet has the same direction as impact_velocity_

        Scalar bullet_cylinder_length_;//default: 0.0, the cylinder length of bullet
        Scalar recovery_coefficient_;  //default: 0.0,  
                                       //0.0 means perfect inelasticity, while 1.0 means perfect elasticity.
                                       //only valid for normal relative velocity
        bool enable_project_pos_;      //default: true, whether project the particle out of sphere

    };


} // end of namespace Physika
#endif // PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_IMPACT_METHODS_PDM_IMPACT_METHOD_BULLET_H