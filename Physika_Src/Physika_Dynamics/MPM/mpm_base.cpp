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

#include "Physika_Core/Grid_Weight_Functions/grid_cubic_weight_functions.h"
#include "Physika_Dynamics/MPM/mpm_base.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMBase<Scalar,Dim>::MPMBase()
    :DriverBase<Scalar>(), weight_function_(NULL)
{
    setWeightFunction<GridPiecewiseCubicSpline<Scalar,Dim>>(); //default weight function is piece-wise cubic b spline
}

template <typename Scalar, int Dim>
MPMBase<Scalar,Dim>::MPMBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :DriverBase<Scalar>(start_frame,end_frame,frame_rate,max_dt,write_to_file), weight_function_(NULL)
{
    setWeightFunction<GridPiecewiseCubicSpline<Scalar,Dim>>(); //default weight function is piece-wise cubic b spline
}

template <typename Scalar, int Dim>
MPMBase<Scalar,Dim>::~MPMBase()
{
    if(weight_function_)
        delete weight_function_;
}

//explicit instantiations
template class MPMBase<float,2>;
template class MPMBase<float,3>;
template class MPMBase<double,2>;
template class MPMBase<double,3>;

}  //end of namespace Physika
