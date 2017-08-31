/*
 * @file driver_plugin_base.h 
 * @Basic class for plugins of a simulation driver.
 * @author Tianxiang Zhang, Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_DYNAMICS_DRIVER_DRIVER_PLUGIN_BASE_H_
#define PHYSIKA_DYNAMICS_DRIVER_DRIVER_PLUGIN_BASE_H_

namespace Physika{

template <typename Scalar> class DriverBase;

template <typename Scalar>
class DriverPluginBase
{
public:
    //constructors && deconstructors
    DriverPluginBase();
    virtual ~DriverPluginBase();

    //functions called in driver
    virtual void onBeginFrame(unsigned int frame) = 0;
    virtual void onEndFrame(unsigned int frame) = 0;
    virtual void onBeginTimeStep(Scalar time, Scalar dt) = 0;//time is current time point, dt is the time step about to begin
    virtual void onEndTimeStep(Scalar time, Scalar dt) = 0; //time is current time point, dt is the time step of ended step

    //basic function
    virtual DriverBase<Scalar>* driver();
    virtual void setDriver(DriverBase<Scalar>* driver) = 0;//be sure to type-check driver in implementation.

protected:
    DriverBase<Scalar>* driver_;
};

}  //end of namespace Physika

#endif  //PHYSIKA_DYNAMICS_DRIVER_DRIVER_PLUGIN_BASE_H_
