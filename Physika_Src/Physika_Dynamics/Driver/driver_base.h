/*
 * @file driver_base.h 
 * @brief Base class of all driver classes. A driver class manages the simulation process.
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

#ifndef PHYSIKA_DYNAMICS_DRIVER_DRIVER_BASE_H_
#define PHYSIKA_DYNAMICS_DRIVER_DRIVER_BASE_H_

#include <string>
#include "Physika_Core/Utilities/Timer/timer.h"

namespace Physika{

template <typename Scalar>
class DriverBase
{
public:
    //callbacks, allow customization during simulation
    //derive a subclass and implement the callback methods
    struct CallBacks
    {
        virtual void beginFrame(int frame)=0;
        virtual void endFrame(int frame)=0;
        virtual void beginTimeStep(Scalar time)=0;
        virtual void endTimeStep(Scalar time,Scalar dt)=0;
        virtual void writeOutput(int frame)=0;
        virtual void restart(int frame)=0;
    };

public:
    DriverBase();
    DriverBase(int start_frame, int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
    DriverBase(int start_frame, int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file, CallBacks *call_backs);
    virtual ~DriverBase();
    virtual void run();//run the simulation from start frame to end frame
    virtual void advanceFrame();//advance one frame
    virtual void initialize()=0;//initialize before the simulation
    virtual void advanceStep(Scalar dt)=0;//advance one time step
    virtual Scalar computeTimeStep()=0;//compute time step with respect to simulation specific conditions
    virtual void write(const char *file_name)=0;//write simulation data to file
    virtual void read(const char *file_name)=0;//read simulation data from file

    inline void setCallBacks(CallBacks *call_backs){call_backs_ = call_backs;}
    inline void setMaxDt(Scalar max_dt){max_dt_ = max_dt;}
    inline Scalar maxDt(){return max_dt_;}
    inline void setFrameRate(Scalar frame_rate){frame_rate_ = frame_rate;}
    inline Scalar frameRate(){return frame_rate_;}
    inline void setStartFrame(int start_frame){start_frame_ = start_frame;}
    inline int getStartFrame(){return start_frame_;}
    inline void setEndFrame(int end_frame){end_frame_ = end_frame;}
    inline int getEndFrame(){return end_frame_;}
    inline void enableWriteToFile(){write_to_file_ = true;}
    inline void disableWriteToFile(){write_to_file_ = false;}
    inline void enableTimer(){enable_timer_=true;}
    inline void disableTimer(){enable_timer_=false;}

protected:
    int start_frame_;
    int end_frame_;
    int restart_frame_;
    Scalar frame_rate_;
    Scalar max_dt_;
    bool write_to_file_;
    CallBacks *call_backs_;
    bool enable_timer_;
    Timer timer_;
    Scalar time_;//current time point of simulation
};

}  //end of namespace Physika

#endif  //PHYSIKA_DYNAMICS_DRIVER_DRIVER_BASE_H_

