/*
 * @file PDM_plugin_base.h 
 * @brief base class of plugins for PDM drivers.
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

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_PLUGINS_PDM_PLUGIN_BASE_H
#define PHYSIKA_DYNAMICS_PDM_PDM_PLUGINS_PDM_PLUGIN_BASE_H

#include "Physika_Dynamics/PDM/PDM_base.h"
#include "Physika_Dynamics/Driver/driver_plugin_base.h"

namespace Physika{

template <typename Scalar> class DriverBase;

template <typename Scalar, int Dim>
class PDMPluginBase: public DriverPluginBase<Scalar>
{
public:
	PDMPluginBase();
	virtual ~PDMPluginBase();

	//inherited virtual methods
	virtual void onBeginFrame(unsigned int frame) = 0;
	virtual void onEndFrame(unsigned int frame) = 0;
	virtual void onBeginTimeStep(Scalar time, Scalar dt) = 0;
	virtual void onEndTimeStep(Scalar time, Scalar dt) = 0;

	virtual PDMBase<Scalar, Dim>* driver();
	virtual void setDriver(DriverBase<Scalar>* driver);

protected:

};

} // end of namespace Physika

#endif //PHYSIKA_DYNAMICS_PDM_PDM_PLUGINS_PDM_PLUGIN_BASE_H