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

#include <iostream>
#include <string>
#include "Physika_Core/Utilities/Timer/timer.h"

namespace Physika{

template <typename Scalar>
class DriverBase
{
public:
    DriverBase();
    DriverBase(int start_frame, int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file, CALLBACKS *call_backs=NULL);
    virtual ~DriverBase();
    virtual void run();//run the simulation from start frame to end frame
    virtual void advanceFrame();//advance one frame
    virtual void initialize()=0;//initialize before the simulation
    virtual void advanceStep(Scalar dt)=0;//advance one time step
    virtual Scalar computeTimeStep()=0;//compute time step with respect to simulation specific conditions
    virtual void write(const char *file_name)=0;//write simulation data to file
    virtual void read(const char *file_name)=0;//read simulation data from file

    inline void setCallBacks(CALLBACKS *call_backs){call_backs_=call_backs;}
    inline void setMaxDt(Scalar max_dt){max_dt_ = max_dt;}
    inline Scalar maxDt(){return max_dt_;}
    inline void setFrameRate(Scalar frame_rate){frame_rate_ = frame_rate;}
    inline Scalar frameRate(){return frame_rate_;}
    inline void setStartFrame(int start_frame){start_frame_ = start_frame;}
    inline int getStartFrame(){return start_frame_;}
    inline void setEndFrame(int end_frame){end_frame_ = end_frame;}
    inline int getEndFrame(){return end_frame_;}
    inline void enableWriteToFile(){write_to_file_=true;}
    inline void disableWriteToFile(){write_to_file=false;}
    inline void enableTimer(){enable_timer_=true;}
    inline void disableTimer(){enable_timer_=false;}
public:
    //callbacks, allow customization during simulation
    struct CALLBACKS
    {
	virtual void beginFrame(int frame)=0;
	virtual void endFrame(int frame)=0;
	virtual void beginTimeStep(Scalar time)=0;
	virtual void endTimeStep(Scalar time,Scalar dt)=0;
	virtual void writeOutput(int frame)=0;
	virtual void restart(int frame)=0;
    };
protected:
    int start_frame_;
    int end_frame_;
    int restart_frame_;
    Scalar frame_rate_;
    Scalar max_dt_;
    bool write_to_file_;
    CALLBACKS *call_backs_;
    bool enable_timer_;
    Timer timer_;
    Scalar time_;//current time point of simulation
};

//implementation
template <typename Scalar>
DriverBase<Scalar>::DriverBase()
    :start_frame_(0),end_frame_(0),restart_frame_(-1),frame_rate_(0),
    max_dt_(0),write_to_file_(false),call_backs_(NULL),enable_timer_(true),
    time_(0)
{
}

template <typename Scalar>
DriverBase<Scalar>::DriverBase(int start_frame, int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file, CALLBACKS *call_backs)
    :start_frame_(start_frame),end_frame_(end_frame),restart_frame_(-1),frame_rate_(frame_rate),
    max_dt_(max_dt),write_to_file_(write_to_file),call_backs_(call_backs),enable_timer_(true),
    time_(0)
{
}

template <typename Scalar>
DriverBase<Scalar>::~DriverBase()
{
}

template <typename Scalar>
void DriverBase<Scalar>::run()
{
    initialize();
    for(int frame=start_frame_;frame<=end_frame_;++frame)
    {
	std::cout<<"Begin Frame "<<frame<<"\n";
	if(call_backs_) call_backs_->beginFrame(frame);
	if(enable_timer_) timer_.startTimer();
	advanceFrame();
	std::cout<<"End Frame "<<frame<<" ";
	if(enable_timer_)
	{
	    timer_.stopTimer();
	    std::cout<<timer_.getElapsedTime()<<" s";
	}
	std::cout<<"\n";
	if(call_backs_) call_backs_->endFrame(frame);
	if(write_to_file_)
	{
	    std::string file_name="Frame "+frame;
	    write(file_name.c_str());
	}
    }
}

template <typename Scalar>
void DriverBase<Scalar>::advanceFrame()
{
    Scalar frame_dt=1.0/frame_rate_;
    Scalar initial_time=time_;
    Scalar finish_time=time_+frame_dt;
    bool frame_done=false;
    while(!frame_done)
    {
        //begin time step callbacks
	if(call_backs_) call_backs_->beginTimeStep(time_);
        //compute time step
	Scalar dt=computeTimeStep();
	dt=(dt>max_dt_)?max_dt_:dt;
	//update time and maybe time step
        if(finish_time-time<=dt)
	{
	    dt=finish_time-time;
	    time=finish_time;
	    frame_done=true;
	}
	else
	    time+=dt;
        //advance step
	advanceStep(dt);
        //end time step callbacks
	if(call_backs_) call_backs_->endTimeStep(time_,dt);
    }
}

}  //end of namespace Physika

#endif  //PHYSIKA_DYNAMICS_DRIVER_DRIVER_BASE_H_














