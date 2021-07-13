/*
 * @file timer.h 
 * @brief Cross platform timer class.
 * @author FeiZhu
 * @acknowledge Jernej Barbic, author of VegaFEM
 *
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_TIMER_TIMER_H_
#define PHYSIKA_CORE_TIMER_TIMER_H_

#if (defined __unix__) || (defined __APPLE__)
#include <sys/time.h>
#elif (defined _WIN32)
#include <windows.h>
#endif

namespace PhysIKA {

class CTimer
{
public:
    CTimer();
    ~CTimer();
    void   start();
    void   stop();
    double getElapsedTime();

protected:
#if (defined __unix__) || (defined __APPLE__)
    long start_sec_, stop_sec_, start_micro_sec_, stop_micro_sec_;
#elif (defined _WIN32)
    LARGE_INTEGER timer_frequency_;
    LARGE_INTEGER start_count_, stop_count_;
#endif
};

}  //end of namespace PhysIKA

#endif  //PHYSIKA_CORE_TIMER_TIMER_H_
