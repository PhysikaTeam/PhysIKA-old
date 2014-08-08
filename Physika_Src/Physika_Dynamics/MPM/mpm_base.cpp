/*
 * @file mpm_base.cpp 
 * @Brief base class of all MPM drivers.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Grid_Weight_Functions/grid_cubic_weight_functions.h"
#include "Physika_Dynamics/MPM/MPM_Plugins/mpm_plugin_base.h"
#include "Physika_Dynamics/MPM/mpm_base.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMBase<Scalar,Dim>::MPMBase()
    :DriverBase<Scalar>(), weight_function_(NULL), weight_radius_cell_scale_(1.0), step_method_(NULL),
     cfl_num_(0.5),sound_speed_(340.0),gravity_(9.8)
{
    //default weight function is piece-wise cubic b spline with support domain of 2 cell
    setWeightFunction<GridPiecewiseCubicSpline<Scalar,Dim>>(Vector<Scalar,Dim>(2.0)); 
}

template <typename Scalar, int Dim>
MPMBase<Scalar,Dim>::MPMBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :DriverBase<Scalar>(start_frame,end_frame,frame_rate,max_dt,write_to_file), weight_function_(NULL), weight_radius_cell_scale_(1.0),
     step_method_(NULL),cfl_num_(0.5),sound_speed_(340.0),gravity_(9.8)
{
    //default weight function is piece-wise cubic b spline with support domain of 2 cell
    setWeightFunction<GridPiecewiseCubicSpline<Scalar,Dim>>(Vector<Scalar,Dim>(2.0)); 
}

template <typename Scalar, int Dim>
MPMBase<Scalar,Dim>::~MPMBase()
{
    if(weight_function_)
        delete weight_function_;
    if(step_method_)
        delete step_method_;
}

template <typename Scalar, int Dim>
Scalar MPMBase<Scalar,Dim>::computeTimeStep()
{
    Scalar min_cell_size = minCellEdgeLength();
    Scalar max_particle_vel = maxParticleVelocityNorm();
    this->dt_ = (this->cfl_num_ * min_cell_size)/(this->sound_speed_+max_particle_vel);
    this->dt_ = this->dt_ > this->max_dt_ ? this->max_dt_ : this->dt_;
    return this->dt_;
}

template <typename Scalar, int Dim>
void MPMBase<Scalar,Dim>::advanceStep(Scalar dt)
{
    //plugin operation, begin time step
    MPMPluginBase<Scalar,Dim> *plugin = NULL;
    for(unsigned int i = 0; i < this->plugins_.size(); ++i)
    {
        plugin = dynamic_cast<MPMPluginBase<Scalar,Dim>*>(this->plugins_[i]);
        if(plugin)
            plugin->onBeginTimeStep(this->time_,dt);
    }

    PHYSIKA_ASSERT(this->step_method_);
    this->step_method_->advanceStep(dt);

    //plugin operation, end time step
    for(unsigned int i = 0; i < this->plugins_.size(); ++i)
    {
        plugin = dynamic_cast<MPMPluginBase<Scalar,Dim>*>(this->plugins_[i]);
        if(plugin)
            plugin->onEndTimeStep(this->time_,dt);
    }
}

template <typename Scalar, int Dim>
Scalar MPMBase<Scalar,Dim>::cflConstant() const
{
    return cfl_num_;
}

template <typename Scalar, int Dim>
void MPMBase<Scalar,Dim>::setCFLConstant(Scalar cfl)
{
    if(cfl<0)
    {
        std::cerr<<"Warning: Invalid CFL constant specified, use default value (0.5) instead!\n";
        cfl_num_ = 0.5;
    }
    else
        cfl_num_ = cfl;
}

template <typename Scalar, int Dim>
Scalar MPMBase<Scalar,Dim>::soundSpeed() const
{
    return sound_speed_;
}

template <typename Scalar, int Dim>
void MPMBase<Scalar,Dim>::setSoundSpeed(Scalar sound_speed)
{
    if(sound_speed<0)
    {
        std::cerr<<"Warning: Negative sound speed specified, use its absolute value instead!\n";
        sound_speed_ = -sound_speed;
    }
    else
        sound_speed_ = sound_speed;
}

template <typename Scalar, int Dim>
Scalar MPMBase<Scalar,Dim>::gravity() const
{
    return gravity_;
}

template <typename Scalar, int Dim>
void MPMBase<Scalar,Dim>::setGravity(Scalar gravity)
{
    if(gravity<0)
    {
        std::cerr<<"Warning: Negative gravity specified, use its absolute value instead!\n";
        gravity_ = -gravity;
    }
    else
        gravity_ = gravity;
}

//explicit instantiations
template class MPMBase<float,2>;
template class MPMBase<float,3>;
template class MPMBase<double,2>;
template class MPMBase<double,3>;

}  //end of namespace Physika
