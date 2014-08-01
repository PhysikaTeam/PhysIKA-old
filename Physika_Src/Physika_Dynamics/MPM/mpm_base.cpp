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
#include "Physika_Core/Grid_Weight_Functions/grid_cubic_weight_functions.h"
#include "Physika_Dynamics/MPM/mpm_base.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMBase<Scalar,Dim>::MPMBase()
    :DriverBase<Scalar>(), weight_function_(NULL), step_method_(NULL),
     cfl_num_(0.5),sound_speed_(340.0)
{
    setWeightFunction<GridPiecewiseCubicSpline<Scalar,Dim>>(); //default weight function is piece-wise cubic b spline
}

template <typename Scalar, int Dim>
MPMBase<Scalar,Dim>::MPMBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :DriverBase<Scalar>(start_frame,end_frame,frame_rate,max_dt,write_to_file), weight_function_(NULL), step_method_(NULL),cfl_num_(0.5),sound_speed_(340.0)
{
    setWeightFunction<GridPiecewiseCubicSpline<Scalar,Dim>>(); //default weight function is piece-wise cubic b spline
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
        std::cerr<<"Warning: Invalid sound speed specified, use default value (340m/s) instead!\n";
        sound_speed_ = 340.0;
    }
    else
        sound_speed_ = sound_speed;
}

//explicit instantiations
template class MPMBase<float,2>;
template class MPMBase<float,3>;
template class MPMBase<double,2>;
template class MPMBase<double,3>;

}  //end of namespace Physika
