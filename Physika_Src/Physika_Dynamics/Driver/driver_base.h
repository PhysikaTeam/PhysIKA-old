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
#include "Physika_Core/Timer/timer.h"
#include "Physika_Core/Config_File/config_file.h"

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
 *
 * Physika drivers allow for restarting the simulation from any frame, provided that the
 * simulation data of that frame has been written to file in previous simulation.
 * To implement restart support in subclass of DriverBase, you need to:
 * 1. Implement the write() method where the image of driver class in memory is written to file
 * 2. Implement the read() method where the status of driver class is loadded from file
 * 3. Implement the initSimulationData() method, in which the read() method is called to load driver status
 *    of restart frame into memory
 * Note:
 * 1. If you override the default run() in your subclass, be sure to call initSimulationData() at the begining of
 *    your run() method such that restart works properly
 * 2. In default restart only works if you run the simulation by calling run(), if you run simulation through
 *    advanceFrame() or advanceStep(), you need to call initSimulationData() before entering simulation loop
 * 3. Be sure to return correct value in your withRestartSupport() method to inform the user whether restart
 *    is supported in your driver
 * 
 * For users of drivers: call setRestartFrame() before calling run()
 */

template <typename Scalar> class DriverPluginBase;

template <typename Scalar>
class DriverBase
{
public:
    DriverBase();
    DriverBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
    virtual ~DriverBase();

    //virtual methods
    virtual void initConfiguration(const std::string &file_name)=0; //init the configuration (simulation parameters) from file
    virtual void printConfigFileFormat()=0; //print the format of the configuration file needed for this driver
    virtual void initSimulationData()=0; //prepare data for simulation, in default it's called at the begining of run()
    virtual void run();//run the simulation from start frame to end frame
    virtual void advanceFrame(unsigned int frame);//advance one frame
    virtual void advanceStep(Scalar dt)=0;//advance one time step
    virtual Scalar computeTimeStep()=0;//compute time step with respect to simulation specific conditions, return time step
    virtual void addPlugin(DriverPluginBase<Scalar>* plugin) = 0;//add a plugin in this driver, type-check of driver should be done before assignment.

    //restart support
    virtual bool withRestartSupport() const=0;//indicate whether restart is suported in current implementation
    virtual void write(const std::string &file_name)=0;//write simulation data of current status to file
    virtual void read(const std::string &file_name)=0;//read simulation data of current status from file
    inline void setRestartFrame(unsigned int restart_frame){restart_frame_ = restart_frame;} //set the frame to restart from

    //setters && getters
    inline void setMaxDt(Scalar max_dt){max_dt_ = max_dt;}
    inline Scalar maxDt() const{return max_dt_;}
    inline void setFrameRate(Scalar frame_rate){frame_rate_ = frame_rate;}
    inline Scalar frameRate() const{return frame_rate_;}
    inline void setStartFrame(unsigned int start_frame){start_frame_ = start_frame;}
    inline unsigned int getStartFrame() const{return start_frame_;}
    inline void setEndFrame(unsigned int end_frame){end_frame_ = end_frame;}
    inline unsigned int getEndFrame() const{return end_frame_;}
    inline void enableWriteToFile(){write_to_file_ = true;}
    inline void disableWriteToFile(){write_to_file_ = false;}
    inline bool isWriteToFileEnabled() const {return write_to_file_;}
    inline void enableTimer(){enable_timer_=true;}
    inline void disableTimer(){enable_timer_=false;}
    inline bool isTimerEnabled() const {return enable_timer_;}
    inline Scalar currentTime() const {return time_;} //return current time point

protected:
    unsigned int start_frame_;
    unsigned int end_frame_;
    unsigned int restart_frame_;
    Scalar frame_rate_;
    Scalar max_dt_;
    Scalar dt_; //current dt
    bool write_to_file_;
    bool enable_timer_;
    Timer timer_;
    Scalar total_simulation_time_; //the total time spent on simulation, for performance analysis
    Scalar time_;//current time point since simulation starts (from start frame)
    ConfigFile config_parser_; //parser of configuration file

    std::vector<DriverPluginBase<Scalar>* > plugins_;//Plugin vector. All plugins should be added here and called in corresponding functions
};

}  //end of namespace Physika

#endif  //PHYSIKA_DYNAMICS_DRIVER_DRIVER_BASE_H_
