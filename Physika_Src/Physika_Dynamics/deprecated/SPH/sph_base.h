/*
  * @file sph_base.h
  * @Brief class SPHBase
  * @author Wei Chen
  *
  * This file is part of Physika, a versatile physics simulation library.
  * Copyright (C) 2013- Physika Group.
  *
  * This Source Code Form is subject to the terms of the GNU General Public License v2.0.
  * If a copy of the GPL was not distributed with this file, you can obtain one at:
  * http://www.gnu.org/licenses/gpl-2.0.html
  *
  */

#ifndef PHYSIKA_DYNAMICS_SPH_SPH_BASE_H_
#define PHYSIKA_DYNAMICS_SPH_SPH_BASE_H_

#include "Physika_Dynamics/Driver/driver_base.h"

namespace Physika{

/*
    base class of SPH
    need further consideration
*/

template <typename Scalar> class DriverPluginBase;

template<typename Scalar, int Dim>
class SPHBase: public DriverBase<Scalar>
{
public:

    //ctor & dtor
    SPHBase();
    SPHBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
    virtual ~SPHBase();

    //pure virtual functions
    virtual void advanceStep(Scalar dt) = 0;

    //virtual functions with default implementation
    virtual void initConfiguration(const std::string &file_name);
    virtual void printConfigFileFormat();
    virtual void initSimulationData();

    virtual Scalar computeTimeStep();
    virtual void addPlugin(DriverPluginBase<Scalar>* plugin);

    virtual bool withRestartSupport() const;
    virtual void write(const std::string &file_name);
    virtual void read(const std::string &file_name);

};

} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_SPH_SPH_BASE_H_