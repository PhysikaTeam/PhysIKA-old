/*
 * @file modified_CPDI2_update_method.h 
 * @Brief modified version of CPDI2, the velocities of the particle domain corners are 
 *        updated with interpolated delta velocity of grid nodes, then positions of the
 *        domain corners are updated with corner velocities
 *
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

#ifndef PHYSIKA_DYNAMICS_MPM_CPDI_UPDATE_METHODS_MODIFIED_CPDI2_UPDATE_METHOD_H_
#define PHYSIKA_DYNAMICS_MPM_CPDI_UPDATE_METHODS_MODIFIED_CPDI2_UPDATE_METHOD_H_

#include <vector>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Dynamics/MPM/mpm_internal.h"
#include "Physika_Dynamics/MPM/CPDI_Update_Methods/CPDI2_update_method.h"

namespace Physika{

template <typename Scalar, int Dim> class GridWeightFunction;
template <typename ElementType, int Dim> class ArrayND;

/*
 * constructor is made protected to prohibit creating objects
 * with Dim other than 2 and 3
 */

template <typename Scalar, int Dim>
class ModifiedCPDI2UpdateMethod: public CPDI2UpdateMethod<Scalar,Dim>
{
public:
    ModifiedCPDI2UpdateMethod();
    virtual ~ModifiedCPDI2UpdateMethod();
    //overwrite method in CPDI2UpdateMethod
    //first update the velocities of the domain corners with the interpolated velocity delta from grid
    //then update the position of the domain corners with the velocity
    virtual void updateParticleDomain(const std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> > > > &corner_grid_weight_and_gradient,
                                      const std::vector<std::vector<unsigned int> > &corner_grid_pair_num, Scalar dt,
                                      const ArrayND<Vector<Scalar,Dim>,Dim> &grid_velocity_before,
                                      std::vector<std::vector<Vector<Scalar,Dim> > > &corner_velocities);
protected:
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_CPDI_UPDATE_METHODS_MODIFIED_CPDI2_UPDATE_METHOD_H_
