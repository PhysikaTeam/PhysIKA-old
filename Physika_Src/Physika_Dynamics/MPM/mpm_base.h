/*
 * @file mpm_base.h 
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

#ifndef PHYSIKA_DYNAMICS_MPM_MPM_BASE_H_
#define PHYSIKA_DYNAMICS_MPM_MPM_BASE_H_

#include <string>
#include "Physika_Dynamics/Driver/driver_base.h"

namespace Physika{

template<typename Scalar> class DriverPluginBase;

template <typename Scalar, int Dim>
class MPMBase: public DriverBase<Scalar>
{
public:
    MPMBase();
    MPMBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
    virtual ~MPMBase();

    //virtual methods
    virtual void initConfiguration(const std::string &file_name)=0;
    virtual void advanceStep(Scalar dt)=0;
    virtual Scalar computeTimeStep()=0;
    virtual void addPlugin(DriverPluginBase<Scalar> *plugin)=0;
    virtual bool withRestartSupport() const=0;
    virtual void write(const std::string &file_name)=0;
    virtual void read(const std::string &file_name)=0;

protected:
    virtual void initialize()=0;
};

}  //end of namespace Physika

#endif
