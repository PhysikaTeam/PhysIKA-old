/*
 * @file SPH_Base.h 
 * @Basic SPH class,all SPH method inherit from it.
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

#ifndef PHYSIKA_DYNAMICS_SPH_SPH_BASE_H_
#define PHYSIKA_DYNAMICS_SPH_SPH_BASE_H_

#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Arrays/array_manager.h"
#include "Physika_Core/Arrays/array.h"
#include "Physika_Core/Vectors/vector.h"
#include "Physika_Dynamics/Driver/driver_base.h"



namespace Physika{

template <typename Scalar, int Dim>
class SPHBase:public DriverBase<Scalar>
{
public:
    SPHBase();
    virtual ~SPHBase();

    virtual void initialize();
    virtual void initSceneBoundary();

    virtual float getTimeStep(){ return time_step_; }
    virtual void advance(Scalar dt);
    virtual void stepEuler(Scalar dt);
    virtual void computeNeighbors();
    virtual void computeVolume();
    virtual void computeDensity();
    
    void boundaryHandling();

    void saveVelocities(std::string in_path, unsigned int in_iter);
    void savePositions(std::string in_path, unsigned int in_iter);

    Vector<Scalar, Dim> * getPositionPtr() { return position_.data();}
    unsigned int particleNum () { return particle_num_; }


public:
    virtual void allocMemory(unsigned int particle_num);
    
    //Use different buffer instead of the Particle object is more efficient. . 
    //particle attribute need to be rearranged each step;
    Array<unsigned> particle_index_;
    Array<Scalar> mass_;
    Array<Vector<Scalar, Dim>> position_;
    Array<Vector<Scalar, Dim>> velocity_;
    Array<Vector<Scalar, Dim>> normal_;

    Array<Vector<Scalar, Dim>> viscous_force_;
    Array<Vector<Scalar, Dim>> pressure_force_;
    Array<Vector<Scalar, Dim>> surface_force_;

    Array<Scalar> volume_;
    Array<Scalar> pressure_;
    Array<Scalar> density_;

    unsigned int sim_itor_;

    Scalar time_step_;
    Scalar viscosity_;
    Vector<Scalar, Dim> gravity_;
    Scalar surface_tension_;

    Scalar sampling_distance_;
    Scalar smoothing_length_;

    unsigned int particle_num_;

    Scalar reference_density_;

    ArrayManager dataManager_;
};

} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_SPH_SPH_BASE_H_
