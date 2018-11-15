/*
 * @file PDM_step_method_base.h 
 * @Basic PDMStepMethodBase class. basic step method class, a simplest and straightforward explicit step method implemented
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

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_STEP_METHOD_BASE_H
#define PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_STEP_METHOD_BASE_H

#include "Physika_Core/Utilities/dimension_trait.h"

namespace Physika{

template <typename Scalar, int Dim> class PDMBase;
template <typename Scalar, int Dim> class PDMCollisionMethodBase;
template <typename Scalar, int Dim> class PDMFractureMethodBase;
template <typename Scalar, int Dim> class PDMImpactMethodBase;
template <typename Scalar, int Dim> class PDMTopologyControlMethodBase;
template <typename Scalar, int Dim> class PDMBoundaryConditionMethod;
template <typename Scalar, int Dim> class PDMFamily;

template <typename Scalar, int Dim>
class PDMStepMethodBase
{
public:
    PDMStepMethodBase();
    PDMStepMethodBase(PDMBase<Scalar, Dim> * pdm_base);
    virtual ~PDMStepMethodBase();

    // setter & getter
    // note: when step method is added to PDM driver, it will set pdm_base_ automatically
    // so it is unnecessary to set pdm_base_ manually, though you can set it by yourself.
    virtual void setPDMDriver(PDMBase<Scalar, Dim> * pdm_base);

    // Fracture
    void setFractureMethod(PDMFractureMethodBase<Scalar,Dim> * fracture_method);
    void enableFracture();
    void disableFracture();
    bool isFractureEnabled() const;

    // Collision
    void setCollisionMethod(PDMCollisionMethodBase<Scalar,Dim> * collision_method);
    void enableCollision();
    void disableCollision();
    bool isCollisionEnabled() const;

    // Impact
    void setImpactMethod(PDMImpactMethodBase<Scalar,Dim> * impact_method);
    void enableImpact();
    void disableImpact();
    bool isImpactEnabled() const;

    // Topology Control
    void setTopologyControlMethod(PDMTopologyControlMethodBase<Scalar, Dim> * topology_control_method);
    void enableTopologyControl();
    void disableTopologyControl();
    bool isTopologyControlEnabled() const;

    // Boundary Condition
    void setBoundaryConditionMethod(PDMBoundaryConditionMethod<Scalar, Dim> * boundary_condition_method);
    void enableBoundaryCondition();
    void disableBoundaryCondition();
    bool isBoundaryConditionEnabled() const;
    

    PDMBase<Scalar, Dim> * PDMDriver();
    const PDMBase<Scalar, Dim> * PDMDriver() const;

    // dynamically advance one step
    virtual void advanceStep(Scalar dt);

protected:
    // the calcualteForce function will not clear force, which should be done in driver
    virtual void calculateForce(Scalar dt) = 0;

    // update all particle's family parameters (cur_rest_relative_pos et al.)
    virtual void updateParticleFamilyParameters();
    virtual void updateSpecifiedParticleFamilyParameters(unsigned int par_k, PDMFamily<Scalar, Dim> & pdm_family);

    void addGravity();

protected:
    PDMBase<Scalar, Dim> * pdm_base_;
    bool enable_fracture_;
    PDMFractureMethodBase<Scalar, Dim> * fracture_method_;
    bool enable_collision_;
    PDMCollisionMethodBase<Scalar, Dim> * collision_method_;
    bool enable_impact_;
    PDMImpactMethodBase<Scalar, Dim> * impact_method_;
    bool enable_topology_control_;
    PDMTopologyControlMethodBase<Scalar, Dim> * topology_control_method_;
    bool enable_boundary_condition_;
    PDMBoundaryConditionMethod<Scalar, Dim> * boundary_condition_method_;
};


} // end of namespace Physika

#endif //PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_STEP_METHOD_BASE