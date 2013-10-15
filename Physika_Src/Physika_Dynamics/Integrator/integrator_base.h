/*
 * @file integrator_base.h 
 * @brief Base class of integrator, all integrator inherite from this class.
 * @author Sheng Yang
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_DYNAMICS_INTEGRATOR_INTEGRATOR_BASE_H_
#define PHYSIKA_DYNAMICS_INTEGRATOR_INTEGRATOR_BASE_H_

namespace Physika{

	class Integrator_Base
	{
	public:
		Integrator_Base();
		~Integrator_Base();
	protected:
	};

} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_INTEGRATOR_INTEGRATOR_BASE_H_
