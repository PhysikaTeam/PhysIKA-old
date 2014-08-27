/*
 * @file CPDI2_update_method.cpp 
 * @Brief the particle domain update procedure introduced in paper:
 *        "Second-order convected particle domain interpolation with enrichment for weak
 *         discontinuities at material interfaces"
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

#include "Physika_Core/Arrays/array_Nd.h"
#include "Physika_Core/Grid_Weight_Functions/grid_weight_function.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Dynamics/MPM/CPDI_mpm_solid.h"
#include "Physika_Dynamics/MPM/CPDI_Update_Methods/CPDI2_update_method.h"

namespace Physika{

template <typename Scalar>
CPDI2UpdateMethod<Scalar,2>::CPDI2UpdateMethod()
    :CPDIUpdateMethod<Scalar,2>()
{
}

template <typename Scalar>
CPDI2UpdateMethod<Scalar,2>::~CPDI2UpdateMethod()
{
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,2>::updateParticleDomain(const std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > &particle_grid_weight_and_gradient,
                                                         const std::vector<unsigned int> &particle_grid_pair_num)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    ArrayND<Vector<Scalar,2>,2> particle_domain;
    const std::vector<SolidParticle<Scalar,2>*> &particles = this->cpdi_driver_->allParticles();
    for(unsigned int i = 0; i < particles.size(); ++i)
    {
        this->cpdi_driver_->currentParticleDomain(i,particle_domain);
        for(typename ArrayND<Vector<Scalar,2>,2>::Iterator iter = particle_domain.begin(); iter != particle_domain.end(); ++iter)
        {
            Vector<Scalar,2> cur_corner_pos = *iter;
        }
    }
//TO DO
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,2>::updateParticleInterpolationWeight(unsigned int particle_idx, const GridWeightFunction<Scalar,2> &weight_function,
                                           std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > &particle_grid_weight_and_gradient,
                                           unsigned int &particle_grid_pair_num)
{
//TO DO
}

///////////////////////////////////////////////////// 3D ///////////////////////////////////////////////////

template <typename Scalar>
CPDI2UpdateMethod<Scalar,3>::CPDI2UpdateMethod()
    :CPDIUpdateMethod<Scalar,3>()
{
}

template <typename Scalar>
CPDI2UpdateMethod<Scalar,3>::~CPDI2UpdateMethod()
{
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,3>::updateParticleDomain(const std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > &particle_grid_weight_and_gradient,
                                                         const std::vector<unsigned int> &particle_grid_pair_num)
{
//TO DO
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,3>::updateParticleInterpolationWeight(unsigned int particle_idx, const GridWeightFunction<Scalar,3> &weight_function,
                                           std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > &particle_grid_weight_and_gradient,
                                           unsigned int &particle_grid_pair_num)
{
//TO DO
}

//explicit instantiations
template class CPDI2UpdateMethod<float,2>;
template class CPDI2UpdateMethod<double,2>;
template class CPDI2UpdateMethod<float,3>;
template class CPDI2UpdateMethod<double,3>;

}  //end of namespace Physika
