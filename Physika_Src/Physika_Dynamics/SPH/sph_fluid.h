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
#include "Physika_Dynamics/sph/sph_neighbor_query.h"

namespace Physika{

template <typename Scalar, int Dim>
class SPHFluid:public SPHBase<Scalar, Dim>
{
public:
    SPHFluid();
    ~SPHFluid();

    virtual void initialize();
    virtual void initSceneBoundary();

    virtual float getTimeStep();
    virtual void advance(Scalar dt);
    virtual void stepEuler(Scalar dt);
    virtual void computeNeighbors();
    virtual void computeVolume();
    virtual void computeDensity();

    virtual void computeSurfaceTension();
    virtual void allocMemory(unsigned int particle_num);


    void computePressure(Scalar dt);
    void computePressureForce(Scalar dt);
    void computeViscousForce(Scalar dt);

    void advect(Scalar dt);

protected:

    Scalar max_mass_;
    Scalar min_mass_;

    Scalar max_length_;
    Scalar min_length_;

    Array<Scalar> phi_;
    Array<Scalar> energey_;

    Array<NeighborList<Scalar>> neighborLists_;
    
};

} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_SPH_SPH_FLUID_H_
