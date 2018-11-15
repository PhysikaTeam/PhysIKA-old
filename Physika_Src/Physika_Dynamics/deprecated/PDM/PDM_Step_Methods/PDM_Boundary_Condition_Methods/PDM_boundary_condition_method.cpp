/*
 * @file PDM_boundary_condition_method.cpp 
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

#include <string>
#include <fstream>

#include "Physika_Dynamics/PDM/PDM_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Boundary_Condition_Methods/PDM_boundary_condition_method.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMBoundaryConditionMethod<Scalar, Dim>::PDMBoundaryConditionMethod()
    :pdm_base_(NULL), cancel_boundary_condition_time_step_(-1), 
    floor_pos_(0.0), enable_floor_boundary_condition_(false),
    enable_floor_reset_particle_pos_(false), enable_floor_reset_mesh_vert_pos_(false),
    floor_normal_recovery_coefficient_(1.0), floor_tangent_recovery_coefficient_(1.0),
    enable_compression_boundary_condition_(false), compression_tangent_recovery_coefficient_(1.0),
    up_plane_pos_(0.0), down_plane_pos_(0.0), up_plane_vel_(0.0), down_plane_vel_(0.0)
{

}

template <typename Scalar, int Dim>
PDMBoundaryConditionMethod<Scalar, Dim>::~PDMBoundaryConditionMethod()
{

}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::setDriver(PDMBase<Scalar, Dim> * pdm_base)
{
    if (pdm_base == NULL)
    {
        std::cerr<<"error: can't set NULL driver to Impact Method!\n";
        std::exit(EXIT_FAILURE);
    }
    this->pdm_base_ = pdm_base;
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::addFixParticle(unsigned int par_id)
{
    this->reset_pos_vec_.push_back(par_id);
    this->reset_vel_vec_.push_back(par_id);
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::addResetVelParticle(unsigned int par_id)
{
    this->reset_vel_vec_.push_back(par_id);
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::addSetParticleVel(const std::vector<unsigned int> & par_id_vec, const Vector<Scalar, Dim> & par_vel)
{
    PHYSIKA_ASSERT(this->set_vel_par_vec_.size() == this->set_vel_vec_.size());
    this->set_vel_par_vec_.push_back(par_id_vec);
    this->set_vel_vec_.push_back(par_vel);
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::addSetParticleVel(unsigned int par_id, const Vector<Scalar, Dim> & par_vel)
{
    PHYSIKA_ASSERT(this->set_vel_par_vec_.size() == this->set_vel_vec_.size());
    std::vector<unsigned int> par_id_vec;
    par_id_vec.push_back(par_id);
    this->set_vel_par_vec_.push_back(par_id_vec);
    this->set_vel_vec_.push_back(par_vel);
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::addParticleExtraForce(const std::vector<unsigned int> & par_id_vec, const Vector<Scalar, Dim> & extra_force)
{
    PHYSIKA_ASSERT(this->extra_force_par_vec_.size() == this->extra_force_vec_.size());
    this->extra_force_par_vec_.push_back(par_id_vec);
    this->extra_force_vec_.push_back(extra_force);
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::addParticleExtraForce(unsigned int par_id, const Vector<Scalar, Dim> & extra_force)
{
    PHYSIKA_ASSERT(this->extra_force_par_vec_.size() == this->extra_force_vec_.size());
    std::vector<unsigned int> par_id_vec;
    par_id_vec.push_back(par_id);
    this->extra_force_par_vec_.push_back(par_id_vec);
    this->extra_force_vec_.push_back(extra_force);
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::addResetXVelParticle(unsigned int par_id)
{
    this->reset_x_vel_vec_.push_back(par_id);
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::addResetYVelParticle(unsigned int par_id)
{
    this->reset_y_vel_vec_.push_back(par_id);
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::addResetZVelParticle(unsigned int par_id)
{
    this->reset_z_vel_vec_.push_back(par_id);
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::setFloorPos(Scalar floor_pos)
{
    this->floor_pos_ = floor_pos;
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::enableFloorBoundaryCondition()
{
    this->enable_floor_boundary_condition_= true;
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::disableFloorBoundaryCondition()
{
    this->enable_floor_boundary_condition_ = false;
}

template <typename Scalar, int Dim>
bool PDMBoundaryConditionMethod<Scalar, Dim>::isEnableFloorBoundaryCondition() const
{
    return this->enable_floor_boundary_condition_;
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::enableFloorResetParticlePos()
{
    this->enable_floor_reset_particle_pos_ = true;
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::disableFloorResetParticlePos()
{
    this->enable_floor_reset_particle_pos_ = false;
}

template <typename Scalar, int Dim>
bool PDMBoundaryConditionMethod<Scalar, Dim>::isEnableFloorResetParticlePos() const
{
    return this->enable_floor_reset_particle_pos_;
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::enableFloorResetMeshVertPos()
{
    this->enable_floor_reset_mesh_vert_pos_ = true;
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::disableFloorResetMeshVertPos()
{
    this->enable_floor_reset_mesh_vert_pos_ = false;
}

template <typename Scalar, int Dim>
bool PDMBoundaryConditionMethod<Scalar, Dim>::isEnableFloorResetMeshVertPos() const
{
    return this->enable_floor_reset_mesh_vert_pos_;
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::setFloorNormalRecoveryCoefficient(Scalar floor_normal_recovery_coefficient)
{
    this->floor_normal_recovery_coefficient_ = floor_normal_recovery_coefficient;
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::setFloorTangentRecoveryCoefficient(Scalar floor_tangent_recovery_coefficient)
{
    this->floor_tangent_recovery_coefficient_ = floor_tangent_recovery_coefficient;
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::enableCompressionBoundaryCondition()
{
    this->enable_compression_boundary_condition_ = true;
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::disableCompressionBoundaryCondition()
{
    this->enable_compression_boundary_condition_ = false;
}

template <typename Scalar, int Dim>
bool PDMBoundaryConditionMethod<Scalar, Dim>::isEnableCompressionBoundaryCondition() const
{
    return this->enable_compression_boundary_condition_;
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::setUpPlanePos(Scalar up_plane_pos)
{
    this->up_plane_pos_ = up_plane_pos;
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::setDownPlanePos(Scalar down_plane_pos)
{
    this->down_plane_pos_ = down_plane_pos;
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::setUpPlaneVel(Scalar up_plane_vel)
{
    this->up_plane_vel_ = up_plane_vel;
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::setDownPlaneVel(Scalar down_plane_vel)
{
    this->down_plane_vel_ = down_plane_vel;
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::exertFloorBoundaryCondition()
{
    ////need further consideration
    //////////////////////////////////////////////////////////////////////////////////////////
    std::fstream file("floor.txt", std::ios::in);
    if (file.fail() == false)
    {
        std::string parameter_name;
        file>>parameter_name>>this->enable_floor_boundary_condition_;    
        file>>parameter_name>>this->floor_pos_;
        if (file.good()) file>>parameter_name>>this->enable_floor_reset_particle_pos_;
        if (file.good()) file>>parameter_name>>this->enable_floor_reset_mesh_vert_pos_;
        if (file.good()) file>>parameter_name>>this->floor_normal_recovery_coefficient_;
        if (file.good()) file>>parameter_name>>this->floor_tangent_recovery_coefficient_;
    }
    file.close();
    /////////////////////////////////////////////////////////////////////////////////////////

    if (this->enable_floor_boundary_condition_ == false) return;
    
    unsigned int contact_floor_num = 0;

    #pragma omp parallel for
    for (long long par_idx = 0; par_idx < this->pdm_base_->numSimParticles(); par_idx++)
    {
        const Vector<Scalar, Dim> & par_pos = this->pdm_base_->particleCurrentPosition(par_idx);
        const Vector<Scalar, Dim> & par_vel = this->pdm_base_->particleVelocity(par_idx);

        if (par_pos[1] < this->floor_pos_)
        {
            #pragma omp atomic
            contact_floor_num ++;

            //set new particle pos, need further consideration
            if (this->enable_floor_reset_particle_pos_ == true)
            {
                Vector<Scalar, Dim> new_par_pos = par_pos;
                new_par_pos[1] = this->floor_pos_;
                this->pdm_base_->setParticleCurPos(par_idx, new_par_pos);
            }
           
            //set new particle velocity
            Vector<Scalar, Dim> new_par_vel(0.0);
            for (unsigned int i = 0; i < Dim; i++)
            {
                if (i == 1 && par_vel[1] < 0.0) 
                    new_par_vel[i] = -this->floor_normal_recovery_coefficient_*par_vel[i];
                else        
                    new_par_vel[i] =  this->floor_tangent_recovery_coefficient_*par_vel[i];
            }
            this->pdm_base_->setParticleVelocity(par_idx, new_par_vel);
        }
    }

    if (this->enable_floor_reset_mesh_vert_pos_ == true)
    {
        VolumetricMesh<Scalar, Dim> * mesh = this->pdm_base_->mesh();
        #pragma omp parallel for
        for(long long vert_idx = 0; vert_idx < mesh->vertNum(); vert_idx++)
        {
            Vector<Scalar, Dim> vert_pos = mesh->vertPos(vert_idx);
            if (vert_pos[1] < this->floor_pos_)
            {
                vert_pos[1] = this->floor_pos_;
                mesh->setVertPos(vert_idx, vert_pos);
            }
        }
    }
    

    std::cout<<"--------------------------------------------------------------------"<<std::endl;
    std::cout<<"enable_floor_boundary_condition: "<<this->enable_floor_boundary_condition_<<std::endl;
    std::cout<<"floor_pos: "<<this->floor_pos_<<std::endl;
    std::cout<<"enable_floor_reset_particle_pos: "<<this->enable_floor_reset_particle_pos_<<std::endl;
    std::cout<<"enable_floor_reset_mesh_vert_pos: "<<this->enable_floor_reset_mesh_vert_pos_<<std::endl;
    std::cout<<"floor_normal_recovery_coefficient: "<<this->floor_normal_recovery_coefficient_<<std::endl;
    std::cout<<"floor_tangent_recovery_coefficient: "<<this->floor_tangent_recovery_coefficient_<<std::endl;
    std::cout<<"contact_floor_num: "<<contact_floor_num<<std::endl;
    std::cout<<"--------------------------------------------------------------------"<<std::endl;

}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::exertCompressionBoundaryCondition(Scalar dt)
{
    ////need further consideration
    //////////////////////////////////////////////////////////////////////////////////////////
    std::fstream file("compression_plane.txt", std::ios::in);
    if (file.fail() == false)
    {
        std::string parameter_name;
        
        Scalar up_plane_pos;
        Scalar down_plane_pos;
        unsigned int convert_step;
        Scalar first_up_plane_vel;
        Scalar second_up_plane_vel;
        Scalar first_down_plane_vel;
        Scalar second_down_plane_vel;
        Scalar compression_tangent_recovery_coefficient;

        file>>parameter_name>>this->enable_compression_boundary_condition_;    
        file>>parameter_name>>up_plane_pos;
        file>>parameter_name>>down_plane_pos;
        file>>parameter_name>>convert_step;
        file>>parameter_name>>first_up_plane_vel;
        file>>parameter_name>>second_up_plane_vel;
        file>>parameter_name>>first_down_plane_vel;
        file>>parameter_name>>second_down_plane_vel;
        file>>parameter_name>>compression_tangent_recovery_coefficient;

        if (this->pdm_base_->timeStepId() == 0)
        {
            this->up_plane_pos_ = up_plane_pos;
            this->down_plane_pos_ = down_plane_pos;
        }

        if (this->pdm_base_->timeStepId() <= convert_step)
        {
            this->up_plane_vel_ = first_up_plane_vel;
            this->down_plane_vel_ = first_down_plane_vel;
        }
        else
        {
            this->up_plane_vel_ = second_up_plane_vel;
            this->down_plane_vel_ = second_down_plane_vel;
        }

        this->compression_tangent_recovery_coefficient_ = compression_tangent_recovery_coefficient;

        std::cout<<"=============================================================================\n";
        std::cout<<"up_plane_pos: "<<this->up_plane_pos_<<std::endl;
        std::cout<<"down_plane_pos: "<<this->down_plane_pos_<<std::endl;
        std::cout<<"up_plane_vel: "<<this->up_plane_vel_<<std::endl;
        std::cout<<"down_plane_vel: "<<this->down_plane_vel_<<std::endl;
        std::cout<<"up_plane_particle_set_size: "<<this->up_plane_particle_set_.size()<<std::endl;
        std::cout<<"down_plane_particle_set_size: "<<this->down_plane_particle_set_.size()<<std::endl;
        std::cout<<"compression_tangent_recovery_coefficient: "<<this->compression_tangent_recovery_coefficient_<<std::endl;
        std::cout<<"=============================================================================\n";
    }
    file.close();
    /////////////////////////////////////////////////////////////////////////////////////////

    if (this->enable_compression_boundary_condition_ == false) return;

    //update up and down plane pos
    this->up_plane_pos_ += this->up_plane_vel_*dt;
    this->down_plane_pos_ += this->down_plane_vel_*dt;

    //update up and down particle set
    #pragma omp parallel for
    for (long long par_idx = 0; par_idx < this->pdm_base_->numSimParticles(); par_idx++)
    {
        const Vector<Scalar, Dim> & par_pos = this->pdm_base_->particleCurrentPosition(par_idx);

        if (par_pos[1] > this->up_plane_pos_)
        {
            #pragma omp critical(INSERT_TO_UP_PLANE)
            this->up_plane_particle_set_.insert(par_idx);
        }

        if (par_pos[1] < this->down_plane_pos_)
        {
            #pragma omp critical(INSERT_TO_DOWN_PLANE)
            this->down_plane_particle_set_.insert(par_idx);
        }

    }

    //exert boundary condition
    for (std::set<unsigned int>::iterator iter = this->up_plane_particle_set_.begin(); iter != this->up_plane_particle_set_.end(); iter++)
    {
        Vector<Scalar, Dim> new_par_vel = this->compression_tangent_recovery_coefficient_*this->pdm_base_->particleVelocity(*iter);
        new_par_vel[1] = this->up_plane_vel_;
        this->pdm_base_->setParticleVelocity(*iter, new_par_vel);
    }

    for (std::set<unsigned int>::iterator iter = this->down_plane_particle_set_.begin(); iter != this->down_plane_particle_set_.end(); iter++)
    {
        Vector<Scalar, Dim> new_par_vel = this->compression_tangent_recovery_coefficient_*this->pdm_base_->particleVelocity(*iter);
        new_par_vel[1] = this->down_plane_vel_;
        this->pdm_base_->setParticleVelocity(*iter, new_par_vel);
    }

}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::setCancelBoundaryConditionTimeStep(int cancel_boundary_condition_time_step)
{
    this->cancel_boundary_condition_time_step_ = cancel_boundary_condition_time_step;
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::resetSpecifiedParticlePos()
{
    //cancel boundary condition, need further consideration
    this->cancelBoundaryConditionTimeStep();

    for (unsigned int i = 0; i < this->reset_pos_vec_.size(); i++)
        this->pdm_base_->setParticleDisplacement(this->reset_pos_vec_[i], Vector<Scalar, Dim>(0.0));
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::resetSpecifiedParticleVel()
{
    //cancel boundary condition, need further consideration
    this->cancelBoundaryConditionTimeStep();

    for (unsigned int i = 0; i < this->reset_vel_vec_.size(); i++)
        this->pdm_base_->setParticleVelocity(this->reset_vel_vec_[i], Vector<Scalar, Dim>(0.0));
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::setSpecifiedParticleVel()
{
    PHYSIKA_ASSERT(this->set_vel_par_vec_.size() == this->set_vel_vec_.size());

    //cancel boundary condition, need further consideration
    this->cancelBoundaryConditionTimeStep();

    for (unsigned int i = 0; i < this->set_vel_par_vec_.size(); i++)
    for (unsigned int j = 0; j < this->set_vel_par_vec_[i].size(); j++)
        this->pdm_base_->setParticleVelocity(this->set_vel_par_vec_[i][j], this->set_vel_vec_[i]);
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::resetSpecifiedParticleXYZVel()
{
    //cancel boundary condition, need further consideration
    this->cancelBoundaryConditionTimeStep();

    for (unsigned int i = 0; i < this->reset_x_vel_vec_.size(); i++)
    {
        unsigned int par_id = this->reset_x_vel_vec_[i];
        Vector<Scalar, Dim> new_vel = this->pdm_base_->particleVelocity(par_id);
        new_vel[0] = 0.0;
        this->pdm_base_->setParticleVelocity(par_id, new_vel);
    }

    for (unsigned int i = 0; i < this->reset_y_vel_vec_.size(); i++)
    {
        unsigned int par_id = this->reset_y_vel_vec_[i];
        Vector<Scalar, Dim> new_vel = this->pdm_base_->particleVelocity(par_id);
        new_vel[1] = 0.0;
        this->pdm_base_->setParticleVelocity(par_id, new_vel);
    }
    
    for (unsigned int i = 0; i < this->reset_z_vel_vec_.size(); i++)
    {
        unsigned int par_id = this->reset_z_vel_vec_[i];
        Vector<Scalar, Dim> new_vel = this->pdm_base_->particleVelocity(par_id);
        new_vel[2] = 0.0;
        this->pdm_base_->setParticleVelocity(par_id, new_vel);
    }

}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::setSpecifiedParticleExtraForce()
{
    PHYSIKA_ASSERT(this->extra_force_par_vec_.size() == this->extra_force_vec_.size());

    //the following codes only used for PDM_Demo_Tear_Armadillo
    /////////////////////////////////////////////////////////////////////////////////////////
    std::fstream file("boundary_force.txt", std::ios::in);
    if (file.fail() == false)
    {
        std::string parameter_name;
        Vector<Scalar, Dim> left_hand_force;
        Vector<Scalar, Dim> right_hand_force;
        Vector<Scalar, Dim> left_foot_force;
        Vector<Scalar, Dim> right_foot_force;

        int left_hand_force_cancel_step;
        int right_hand_force_cancel_step;
        int left_foot_force_cancel_step;
        int right_foot_force_cancel_step;

        file>>parameter_name>>left_hand_force[0]>>left_hand_force[1]>>left_hand_force[2];    
        file>>parameter_name>>right_hand_force[0]>>right_hand_force[1]>>right_hand_force[2]; 
        file>>parameter_name>>left_foot_force[0]>>left_foot_force[1]>>left_foot_force[2];    
        file>>parameter_name>>right_foot_force[0]>>right_foot_force[1]>>right_foot_force[2]; 

        file>>parameter_name>>left_hand_force_cancel_step;   std::cout<<parameter_name<<left_hand_force_cancel_step<<std::endl;
        file>>parameter_name>>right_hand_force_cancel_step;  std::cout<<parameter_name<<right_hand_force_cancel_step<<std::endl;
        file>>parameter_name>>left_foot_force_cancel_step;   std::cout<<parameter_name<<left_foot_force_cancel_step<<std::endl;
        file>>parameter_name>>right_foot_force_cancel_step;  std::cout<<parameter_name<<right_foot_force_cancel_step<<std::endl;

        //set force
        this->extra_force_vec_[0] = left_hand_force/this->extra_force_par_vec_[0].size();
        this->extra_force_vec_[1] = right_hand_force/this->extra_force_par_vec_[1].size();
        this->extra_force_vec_[2] = left_foot_force/this->extra_force_par_vec_[2].size();
        this->extra_force_vec_[3] = right_foot_force/this->extra_force_par_vec_[3].size();

        if (this->pdm_base_->timeStepId() >= left_hand_force_cancel_step)   this->extra_force_vec_[0] = Vector<Scalar, Dim>(0.0);
        if (this->pdm_base_->timeStepId() >= right_hand_force_cancel_step)  this->extra_force_vec_[1] = Vector<Scalar, Dim>(0.0);
        if (this->pdm_base_->timeStepId() >= left_foot_force_cancel_step)   this->extra_force_vec_[2] = Vector<Scalar, Dim>(0.0);
        if (this->pdm_base_->timeStepId() >= right_foot_force_cancel_step)  this->extra_force_vec_[3] = Vector<Scalar, Dim>(0.0);

        std::cout<<"left hand force:  "<<this->extra_force_vec_[0]<<std::endl;
        std::cout<<"right hand force: "<<this->extra_force_vec_[1]<<std::endl;
        std::cout<<"left foot force:  "<<this->extra_force_vec_[2]<<std::endl;
        std::cout<<"right foot force: "<<this->extra_force_vec_[3]<<std::endl;

    }
    file.close();
    ////////////////////////////////////////////////////////////////////////////////////////

    //cancel boundary condition, need further consideration
    this->cancelBoundaryConditionTimeStep();

    for (unsigned int i = 0; i < this->extra_force_par_vec_.size(); i++)
    for (unsigned int j = 0; j < this->extra_force_par_vec_[i].size(); j++)
        this->pdm_base_->addParticleForce(this->extra_force_par_vec_[i][j], this->extra_force_vec_[i]);
}

template <typename Scalar, int Dim>
void PDMBoundaryConditionMethod<Scalar, Dim>::cancelBoundaryConditionTimeStep()
{
    //need further consideration
    /////////////////////////////////////////////////////////////////////////////////////////
    std::fstream file("cancel_time_step.txt", std::ios::in);
    if (file.fail() == false)
        file>>this->cancel_boundary_condition_time_step_;
    file.close();
    std::cout<<"cancel time step: "<<this->cancel_boundary_condition_time_step_<<std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////

    if (this->pdm_base_->timeStepId() == this->cancel_boundary_condition_time_step_)
    {
        std::cout<<"NOTE: function specialBoundaryConditionHandling() is triggered!\n";

        //need further consideration
        //this->pdm_base_->resetParticleVelocity();

        //this->reset_pos_vec_.clear();
        //this->reset_vel_vec_.clear();

        for(unsigned int i = 0; i < this->set_vel_par_vec_.size(); i++)
            this->set_vel_par_vec_[i].clear();

        for(unsigned int i = 0; i < this->extra_force_par_vec_.size(); i++)
            this->extra_force_par_vec_[i].clear();
    }
}

//explicit instantiations
template class PDMBoundaryConditionMethod<float,2>;
template class PDMBoundaryConditionMethod<double,2>;
template class PDMBoundaryConditionMethod<float,3>;
template class PDMBoundaryConditionMethod<double,3>;

}// namespace Physika