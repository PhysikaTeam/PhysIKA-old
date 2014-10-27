/*
 * @file modified_CPDI2_update_method.cpp 
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

#include <map>
#include "Physika_Core/Arrays/array_Nd.h"
#include "Physika_Core/Grid_Weight_Functions/grid_weight_function.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Geometry/Cartesian_Grids/grid.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/MPM/Weight_Function_Influence_Iterators/uniform_grid_weight_function_influence_iterator.h"
#include "Physika_Dynamics/MPM/CPDI_mpm_solid.h"
#include "Physika_Dynamics/MPM/CPDI_Update_Methods/modified_CPDI2_update_method.h"

namespace Physika{

template <typename Scalar, int Dim>
ModifiedCPDI2UpdateMethod<Scalar,Dim>::ModifiedCPDI2UpdateMethod()
    :CPDI2UpdateMethod<Scalar,Dim>()
{
}

template <typename Scalar, int Dim>
ModifiedCPDI2UpdateMethod<Scalar,Dim>::~ModifiedCPDI2UpdateMethod()
{
}

template <typename Scalar, int Dim>
void ModifiedCPDI2UpdateMethod<Scalar,Dim>::updateParticleDomain(
    const std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> > > > &corner_grid_weight_and_gradient,
    const std::vector<std::vector<unsigned int> > &corner_grid_pair_num, Scalar dt,
    const ArrayND<Vector<Scalar,Dim>,Dim> &grid_velocity_before,
    std::vector<std::vector<Vector<Scalar,Dim> > > &corner_velocities)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    ArrayND<Vector<Scalar,Dim>,Dim> particle_domain;
    const std::vector<SolidParticle<Scalar,Dim>*> &particles = this->cpdi_driver_->allParticles();
    for(unsigned int i = 0; i < particles.size(); ++i)
    {
        this->cpdi_driver_->currentParticleDomain(i,particle_domain);
        unsigned int corner_idx = 0;
        for(typename ArrayND<Vector<Scalar,Dim>,Dim>::Iterator iter = particle_domain.begin(); iter != particle_domain.end(); ++corner_idx,++iter)
        {
            for(unsigned int j = 0; j < corner_grid_pair_num[i][corner_idx]; ++j)
            {
                Scalar weight = corner_grid_weight_and_gradient[i][corner_idx][j].weight_value_;
                Vector<Scalar,Dim> node_vel = this->cpdi_driver_->gridVelocity(corner_grid_weight_and_gradient[i][corner_idx][j].node_idx_);
                corner_velocities[i][corner_idx] += weight*(node_vel-grid_velocity_before(corner_grid_weight_and_gradient[i][corner_idx][j].node_idx_));
            }
            Vector<Scalar,Dim> cur_corner_pos = *iter;
            cur_corner_pos += corner_velocities[i][corner_idx]*dt;
            *iter = cur_corner_pos;
        }
        this->cpdi_driver_->setCurrentParticleDomain(i,particle_domain);
    }
}

//explicit instantiations
template class ModifiedCPDI2UpdateMethod<float,2>;
template class ModifiedCPDI2UpdateMethod<double,2>;
template class ModifiedCPDI2UpdateMethod<float,3>;
template class ModifiedCPDI2UpdateMethod<double,3>;

}  //end of namespace Physika
