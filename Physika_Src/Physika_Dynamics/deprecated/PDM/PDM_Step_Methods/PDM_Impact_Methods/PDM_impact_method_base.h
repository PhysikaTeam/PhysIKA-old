/*
 * @file PDM_impact_method_base.h 
 * @brief base class of impact method for PDM drivers.
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

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_IMPACT_METHODS_PDM_IMPACT_METHOD_BASE_H
#define PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_IMPACT_METHODS_PDM_IMPACT_METHOD_BASE_H

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

template <typename Scalar, int Dim> class PDMBase;

template <typename Scalar, int Dim>
class PDMImpactMethodBase
{
public:
    PDMImpactMethodBase();
    virtual ~PDMImpactMethodBase();

    void setDriver(PDMBase<Scalar, Dim> * driver);

    void setImpactVelocity(const Vector<Scalar, Dim> & velocity);
    void setImpactPos(const Vector<Scalar, Dim> & pos);
    void setImpactRadius(Scalar radius);

    const Vector<Scalar, Dim> & impactVelocity() const;
    const Vector<Scalar, Dim> & impactPos() const;
    float impactRadius() const;

    virtual void applyImpact(Scalar dt) = 0;

    void enableTriggerSpecialTreatment();
    void disableTriggerSpecialTreatment();
    virtual void triggerSpecialTreatment(); //default: do nothing
protected:
    PDMBase<Scalar, Dim> * driver_;

    Vector<Scalar,Dim> impact_velocity_; //default:0.0
    Scalar impact_radius_;               //default:0.0, projectile radius
    Vector<Scalar,Dim> impact_pos_;      //default:0.0
    bool trigger_special_treatment_;     //default:false
};


} // end of namespace Physika

#endif // PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_IMPACT_METHODS_PDM_IMPACT_METHOD_BASE_H