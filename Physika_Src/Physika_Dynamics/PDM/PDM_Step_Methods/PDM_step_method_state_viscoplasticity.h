/*
 * @file PDM_step_method_state_viscoplasticiy.h 
 * @Basic PDMStepMethodStateViscoPlasticity class.
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

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_STEP_METHOD_STATE_VISCOPLASTICITY_H
#define PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_STEP_METHOD_STATE_VISCOPLASTICITY_H

#include <vector>
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_step_method_state_elasticity.h"

namespace Physika{

template <typename Scalar, int Dim>
class PDMStepMethodStateViscoPlasticity : public PDMStepMethodStateElasticity<Scalar, Dim>
{
public:
    PDMStepMethodStateViscoPlasticity();
    PDMStepMethodStateViscoPlasticity(PDMState<Scalar, Dim> * pdm_base);

    virtual ~PDMStepMethodStateViscoPlasticity();

    virtual void setPDMDriver(PDMBase<Scalar, Dim> * pdm_base);

    void setKd(Scalar Kd);
    void setVelDecayRatio(Scalar vel_decay_ratio);

    void setLaplacianDampingCoefficient(Scalar laplacian_damping_coefficient);
    void setLaplacianDampingIterTimes(unsigned int laplacian_damping_iter_times);

    void setRcp(Scalar Rcp);
    void setEcpLimit(Scalar Ecp_limit);

    void setRelaxTime(Scalar relax_time);
    void setLambda(Scalar lambda);

    void setHomogeneousYieldCriticalVal(Scalar yield_critical_val);
    void setYieldCriticalVal(unsigned int par_id, Scalar yield_critical_val);
    Scalar yieldCriticalVal(unsigned int par_id) const;

    void enablePlasticStatistics();
    void disablePlasticStatistics();
    bool isPlasticStatisticsEnabled() const;

    virtual void advanceStep(Scalar dt);

protected:
    //override calculateForce since dt is needed in function
    virtual void calculateForce(Scalar dt);

    //add air damping
    void addDampingForce();
    //add laplacian damping
    void addLaplacianDamping();

    // a_1 = 0.5*k , a_2 = -5.0/6.0*u  , a_1 + a_2 = a
    void calculateParameters(PDMState<Scalar,2> * pdm_state, unsigned int par_k, Scalar & a_1, Scalar & a_2, Scalar & b, Scalar & c, Scalar & d, DimensionTrait<2> trait);
    void calculateParameters(PDMState<Scalar,3> * pdm_state, unsigned int par_k, Scalar & a_1, Scalar & a_2, Scalar & b, Scalar & c, Scalar & d, DimensionTrait<3> trait);

protected:
    Scalar Kd_;                     // default:0.0, damping force
    Scalar vel_decay_ratio_;        // default:0.0, no decay

    Scalar       laplacian_damping_coefficient_; //default: 0.0
    unsigned int laplacian_damping_iter_times_;  //default: 0

    
    Scalar Rcp_;                    // default:1.0, the parameter is used to model the fact that material would become stronger under "relative" compression situation
    Scalar Ecp_limit_;              // default:1.0, another parameter used to limit the plasticity in absolute "compression" case

    Scalar relax_time_; //default: 0.001
    Scalar lambda_;     //default: 1.0, a_i = lambda*a

    std::vector<Scalar> yield_critical_val_vec_; //default: numeric_limit<Scalar>::max()

    bool enable_plastic_statistics; //default: false
};

}// end of namespace Physika

#endif //PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_STEP_METHOD_STATE_VISCOPLASTICITY_H
