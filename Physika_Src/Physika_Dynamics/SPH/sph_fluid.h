/*
 * @file sph_fluid.h 
 * @Basic SPH_fluid class, basic fluid simulation uses sph.
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

#ifndef PHYSIKA_DYNAMICS_SPH_SPH_FLUID_H_
#define PHYSIKA_DYNAMICS_SPH_SPH_FLUID_H_

#include "Physika_Dynamics/SPH/sph_base.h"
#include "Physika_Dynamics/SPH/sph_neighbor_query.h"
#include <string>

namespace Physika{

template <typename Scalar, int Dim>
class SPHFluid:public SPHBase<Scalar, Dim>
{
public:
    SPHFluid();
    ~SPHFluid();

    virtual void initialize();
    virtual void initConfiguration();
    virtual void initConfiguration(const std::string &file_name);
    virtual void initSceneBoundary();
    void initScene();

    virtual void advanceStep(Scalar dt)//advance one time step
    {

    }
    virtual Scalar computeTimeStep()//compute time step with respect to simulation specific conditions
    {
        return this->getTimeStep();
    }
    virtual bool withRestartSupport() const
    {
        return false;
    }
    virtual void write(const std::string &file_name)//write simulation data to file
    {

    }
    virtual void read(const std::string &file_name)//read simulation data from file
    {

    }
    virtual void addPlugin(DriverPluginBase<Scalar>* plugin)//add a plugin in this driver. Should be redefined in child class because type-check of driver should be done before assignment.
    {

    }


    virtual Scalar getTimeStep();
    virtual void advance(Scalar dt);
    virtual void stepEuler(Scalar dt);
    virtual void computeNeighbors();
    virtual void computeVolume();
    virtual void computeDensity();

    virtual void computeSurfaceTension();
    virtual void allocMemory(unsigned int particle_num);

    void computeMass();
    void computePressure(Scalar dt);
    void computePressureForce(Scalar dt);
    void computeViscousForce(Scalar dt);

    void advect(Scalar dt);

protected:

    bool init_from_file_;
    std::string init_file_name_;
    int x_num_;
    int y_num_;
    int z_num_;
    
    Scalar max_mass_;
    Scalar min_mass_;

    Scalar max_length_;
    Scalar min_length_;

    Array<Scalar> phi_;
    Array<Scalar> energy_;
    Array<bool> small_scale_;
    Array<Scalar> small_density_;

    Array<NeighborList<Scalar>> neighbor_lists_;
    GridQuery<Scalar, Dim>* grid_;
    
};

} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_SPH_SPH_FLUID_H_
