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

namespace Physika{

template <typename Scalar>
class SPHBase
{
public:
	SPHBase();
    virtual ~SPHBase();

    virtual void initialize();
    virtual void initSceneBoundary();
    
    virtual float getTimeStep();
    virtual void advance(Scalar dt);
    virtual void stepEuler(Scalar dt);
    virtual void computeNeighbors();

    void savePositions();

protected:
};

} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_SPH_SPH_BASE_H_