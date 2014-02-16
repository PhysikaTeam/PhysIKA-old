/*
 * @file timer.cpp
 * @brief Cross platform timer class.
 * @author FeiZhu
 * @acknowledge Jernej Barbic, author of VegaFEM 
 *
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Core/Utilities/Timer/timer.h"

namespace Physika{

Timer::Timer()
{
#if (defined _WIN32)
    QueryPerformanceFrequency(&timer_frequency_);
#endif
    startTimer();
}

Timer::~Timer()
{
}

void Timer::startTimer()
{
#if (defined __unix__) || (defined __APPLE__)
    timeval tv;
    gettimeofday(&tv,NULL);
    start_sec_ = tv.tv_sec;
    start_micro_sec_ = tv.tv_usec;
#elif (defined _WIN32)
    QueryPerformanceCounter(&start_count_);
#endif
}

void Timer::stopTimer()
{
#if (defined __unix__) || (defined __APPLE__)
    timeval tv;
    gettimeofday(&tv,NULL);
    stop_sec_ = tv.tv_sec;
    stop_micro_sec_ = tv.tv_usec;
#elif (defined _WIN32)
    QueryPerformanceCounter(&stop_count_);
#endif
}

double Timer::getElapsedTime()
{
#if (defined __unix__) || (defined __APPLE__)
    double elapsed_time = 1.0 * (stop_sec_ - start_sec_) + 1.0e-6 * (stop_micro_sec_ - start_micro_sec_);
    return elapsed_time;
#elif (defined _WIN32)
    double elaspsed_time = static_cast<double>(stop_count_.QuadPart - start_count_.QuadPart) / static_cast<double>(timer_frequency_.QuadPart);
    return elapsed_time;
#endif
}

} // end of namespace Physika
