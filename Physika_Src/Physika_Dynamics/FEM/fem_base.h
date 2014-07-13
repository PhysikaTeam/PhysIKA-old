/*
 * @file fem_base.h 
 * @Brief Base class of FEM drivers, all FEM methods inherit from it.
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

#ifndef PHYSIKA_DYNAMICS_FEM_FEM_BASE_H_
#define PHYSIKA_DYNAMICS_FEM_FEM_BASE_H_

#include <string>
#include "Physika_Core/Config_File/config_file.h"
#include "Physika_Dynamics/Driver/driver_base.h"

namespace Physika{

template <typename Scalar, int Dim> class VolumetricMesh;

/*
 * Base class of FEM drivers.
 * Two ways to set configurations before simulation:
 * 1. Various setters
 * 2. Load configuration from file
 */

template <typename Scalar, int Dim>
class FEMBase: public DriverBase<Scalar>
{
public:
    FEMBase();
    FEMBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
    virtual ~FEMBase();

    virtual void initialize()=0;    
    virtual void advanceStep(Scalar dt)=0;
    virtual void write(const std::string &file_name)=0;
    virtual void read(const std::string &file_name)=0;
    virtual void addPlugin(DriverPluginBase<Scalar> *plugin)=0;
    virtual void initConfiguration(const std::string &file_name)=0; //init configurations for simulation via configration file

    Scalar gravity() const;
    void setGravity(Scalar gravity);
    void loadSimulationMesh(const std::string &file_name); //load the simulation mesh from file
    void setSimulationMesh(const VolumetricMesh<Scalar,Dim> &mesh);  //set the simulation mesh via an external mesh
protected:
    VolumetricMesh<Scalar,Dim> *simulation_mesh_;
    Scalar gravity_;
    ConfigFile config_parser_;
};

}  //end of namespace Physika

#endif  //PHYSIKA_DYNAMICS_FEM_FEM_BASE_H_
