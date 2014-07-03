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
#include <vector>
#include "Physika_Core/Utilities/Timer/timer.h"

namespace Physika{

/*
 * Driver class uses plugin mechanism  to support customization during simulation
 * The user can inherit a subclass of DriverPluginBase and define the customized 
 * methods therein. These methods can be called in methods of driver class.
 * The default virtual methods of driver plugin are called in specific timing during
 * simulation in DriverBase, e.g., onBeginFrame() is called at the begining of each
 * frame. The user could define more methods for their plugin subclasses and call them
 * wherever they want in their driver subclasses.
 * Example usage:
 * Define a plugin subclass which implements the onBeginFrame() method by popping out a 
 * window at the begining of each frame for rendering purpose.
 */

template <typename Scalar> class DriverPluginBase;

template <typename Scalar>
class DriverBase
{
public:


public:
    DriverBase();
    DriverBase(int start_frame, int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
    virtual ~DriverBase();
    virtual void run();//run the simulation from start frame to end frame
    virtual void advanceFrame();//advance one frame
    virtual void initialize()=0;//initialize before the simulation
    virtual void advanceStep(Scalar dt)=0;//advance one time step
    virtual Scalar computeTimeStep()=0;//compute time step with respect to simulation specific conditions
    virtual void write(const char *file_name)=0;//write simulation data to file
    virtual void read(const char *file_name)=0;//read simulation data from file
    virtual void addPlugin(DriverPluginBase<Scalar>* plugin) = 0;//add a plugin in this driver. Should be redefined in child class because type-check of driver should be done before assignment.

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
    bool enable_timer_;
    Timer timer_;
    Scalar time_;//current time point of simulation

    std::vector<DriverPluginBase<Scalar>* > plugins_;//Plugin vector. All plugins should be added here and called in corresponding functions
};

}  //end of namespace Physika

#endif  //PHYSIKA_DYNAMICS_DRIVER_DRIVER_BASE_H_

