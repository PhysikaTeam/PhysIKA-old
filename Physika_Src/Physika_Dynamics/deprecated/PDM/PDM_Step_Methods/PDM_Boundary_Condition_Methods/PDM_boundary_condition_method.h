/*
 * @file PDM_boundary_condition_method.h 
 * @brief Class PDMBoundaryConditionMethod used to impose boundary conditions
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


#ifndef PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_BOUNDARY_CONDITION_METHODS_PDM_BOUNDARY_CONDITION_METHOD_H
#define PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_BOUNDARY_CONDITION_METHODS_PDM_BOUNDARY_CONDITION_METHOD_H

#include <vector>
#include <set>

/*
 * The boundary condition method is used to replace boundary plugin
 */

template <typename Scalar, int Dim> class PDMBase;

namespace Physika{

template<typename Scalar, int Dim>
class PDMBoundaryConditionMethod
{
public:
    PDMBoundaryConditionMethod();
    ~PDMBoundaryConditionMethod();

    void setDriver(PDMBase<Scalar, Dim> * pdm_base);

    //particle to reset pos & vel
    void addFixParticle(unsigned int par_id);
    //particle to reset vel
    void addResetVelParticle(unsigned int par_id);
    //particle to set specified vel
    void addSetParticleVel(const std::vector<unsigned int> & par_id_vec, const Vector<Scalar, Dim> & par_vel);
    void addSetParticleVel(unsigned int par_id, const Vector<Scalar, Dim> & par_vel);
    //particle to add extra force
    void addParticleExtraForce(const std::vector<unsigned int> & par_id_vec, const Vector<Scalar, Dim> & extra_force);
    void addParticleExtraForce(unsigned int par_id, const Vector<Scalar, Dim> & extra_force);

    //reset particles' specified direction velocities
    void addResetXVelParticle(unsigned int par_id);
    void addResetYVelParticle(unsigned int par_id);
    void addResetZVelParticle(unsigned int par_id);

    //set cancel boundary condition time step
    void setCancelBoundaryConditionTimeStep(int cancel_boundary_condition_time_step);

    //set floor 
    void setFloorPos(Scalar floor_pos);
    void enableFloorBoundaryCondition();
    void disableFloorBoundaryCondition();
    bool isEnableFloorBoundaryCondition() const;
    void enableFloorResetParticlePos();
    void disableFloorResetParticlePos();
    bool isEnableFloorResetParticlePos() const;
    void enableFloorResetMeshVertPos();
    void disableFloorResetMeshVertPos();
    bool isEnableFloorResetMeshVertPos() const;
    void setFloorNormalRecoveryCoefficient(Scalar floor_normal_recovery_coefficient);
    void setFloorTangentRecoveryCoefficient(Scalar floor_tangent_recovery_coefficient);

    //set compression
    void enableCompressionBoundaryCondition();
    void disableCompressionBoundaryCondition();
    bool isEnableCompressionBoundaryCondition() const;
    void setUpPlanePos(Scalar up_plane_pos);
    void setDownPlanePos(Scalar down_plane_pos);
    void setUpPlaneVel(Scalar up_plane_vel);
    void setDownPlaneVel(Scalar down_plane_pos);

    //exert floor boundary condition
    void exertFloorBoundaryCondition();

    //exert compression boundary condition
    void exertCompressionBoundaryCondition(Scalar dt);


    //a unified function boundaryConditionMethod is not provided considering the varieties of boundary conditions
    void resetSpecifiedParticlePos();
    void resetSpecifiedParticleVel();
    void resetSpecifiedParticleXYZVel();
    void setSpecifiedParticleVel();
    void setSpecifiedParticleExtraForce();


    virtual void cancelBoundaryConditionTimeStep();

protected:
    PDMBase<Scalar, Dim> * pdm_base_;

    std::vector<unsigned int> reset_pos_vec_;
    std::vector<unsigned int> reset_vel_vec_;

    std::vector<std::vector<unsigned int> > set_vel_par_vec_;
    std::vector<Vector<Scalar, Dim> >       set_vel_vec_;

    std::vector<std::vector<unsigned int> > extra_force_par_vec_;
    std::vector<Vector<Scalar, Dim> >       extra_force_vec_;

    std::vector<unsigned int> reset_x_vel_vec_;
    std::vector<unsigned int> reset_y_vel_vec_;
    std::vector<unsigned int> reset_z_vel_vec_;

    //need further consideration
    int cancel_boundary_condition_time_step_; //initial: -1, time step to cancel boudary condition time step

    //floor boundary conditions
    Scalar floor_pos_;                          //default: 0.0
    bool enable_floor_boundary_condition_;      //default: false
    bool enable_floor_reset_particle_pos_;      //default: false
    bool enable_floor_reset_mesh_vert_pos_;     //default: false
    Scalar floor_normal_recovery_coefficient_;  //default: 1.0
    Scalar floor_tangent_recovery_coefficient_; //default: 1.0

    //need further consideration, only used for compression demo
    bool enable_compression_boundary_condition_;      //default: false
    Scalar up_plane_pos_;                             //default: 0.0
    Scalar down_plane_pos_;                           //default: 0.0
    Scalar up_plane_vel_;                             //default: 0.0
    Scalar down_plane_vel_;                           //default: 0.0
    Scalar compression_tangent_recovery_coefficient_; //default: 1.0
    std::set<unsigned int> up_plane_particle_set_;
    std::set<unsigned int> down_plane_particle_set_;
    
};


} //namespace Physika

#endif //PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_BOUNDARY_CONDITION_METHODS_PDM_BOUNDARY_CONDITION_METHOD_H