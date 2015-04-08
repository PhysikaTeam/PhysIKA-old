/*
 * @file time_stepping_method.h 
 * @Brief enum of time stepping methods, for use by Physika drivers.
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

#ifndef PHYSIKA_DYNAMICS_UTILITIES_TIME_STEPPING_METHOD_H_
#define PHYSIKA_DYNAMICS_UTILITIES_TIME_TEPPING_METHOD_H_

#include <string>
#include "Physika_Core/Utilities/physika_exception.h"

namespace Physika{
   
/*
 * enumeration of different time integration methods
 * Physika drivers can optionally implement all or a subset of
 * theses stepping methods,and define a method to choose between
 * different stepping schemes, e.g.:
 *
 * void setTimeSteppingMethod(TimeSteppingMethod method)
 */
 enum TimeSteppingMethod{
     FORWARD_EULER,
     BACKWARD_EULER,
     VERLET,
     NEWMARK,
     LEAP_FROG,
     RUNGE_KUTTA_4
 };

//a utility method, return the name of a time stepping method
 inline std::string timeSteppingMethodName(TimeSteppingMethod method)
 {
     switch (method)
     {
     case FORWARD_EULER:
         return std::string("FORWARD_EULER");
         break;
     case BACKWARD_EULER:
         return std::string("BACKWARD_EULER");
         break;
     case VERLET:
         return std::string("VERLET");
         break;
     case NEWMARK:
         return std::string("NEWMARK");
         break;
     case LEAP_FROG:
         return std::string("LEAP_FROG");
         break;
     case RUNGE_KUTTA_4:
         return std::string("RUNGE_KUTTA_4");
         break;
     default:
         throw PhysikaException("Undefined time stepping method!");
         break;
     }
 }

} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_UTILITIES_TIME_STEPPING_METHOD_H_
