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

#include <map>
#include "Physika_Core/Arrays/array_Nd.h"
#include "Physika_Core/Grid_Weight_Functions/grid_weight_function.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Geometry/Cartesian_Grids/grid.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/MPM/Weight_Function_Influence_Iterators/uniform_grid_weight_function_influence_iterator.h"
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
void CPDI2UpdateMethod<Scalar,2>::updateParticleInterpolationWeight(const GridWeightFunction<Scalar,2> &weight_function,
                                                   std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > &particle_grid_weight_and_gradient,
                                                   std::vector<unsigned int> &particle_grid_pair_num,
                                                   std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > > &corner_grid_weight_and_gradient,
                                                   std::vector<std::vector<unsigned int> > &corner_grid_pair_num)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    for(unsigned int i = 0; i < this->cpdi_driver_->particleNum(); ++i)
        updateParticleInterpolationWeight(i,weight_function,particle_grid_weight_and_gradient[i],particle_grid_pair_num[i],
                                          corner_grid_weight_and_gradient[i],corner_grid_pair_num[i]);
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,2>::updateParticleDomain(const std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > > &corner_grid_weight_and_gradient,
                                                       const std::vector<std::vector<unsigned int> > &corner_grid_pair_num, Scalar dt)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    ArrayND<Vector<Scalar,2>,2> particle_domain;
    const std::vector<SolidParticle<Scalar,2>*> &particles = this->cpdi_driver_->allParticles();
    for(unsigned int i = 0; i < particles.size(); ++i)
    {
        this->cpdi_driver_->currentParticleDomain(i,particle_domain);
        unsigned int corner_idx = 0;
        for(typename ArrayND<Vector<Scalar,2>,2>::Iterator iter = particle_domain.begin(); iter != particle_domain.end(); ++corner_idx,++iter)
        {
            Vector<Scalar,2> cur_corner_pos = *iter;
            for(unsigned int j = 0; j < corner_grid_pair_num[i][corner_idx]; ++j)
            {
                Scalar weight = corner_grid_weight_and_gradient[i][corner_idx][j].weight_value_;
                Vector<Scalar,2> node_vel = this->cpdi_driver_->gridVelocity(corner_grid_weight_and_gradient[i][corner_idx][j].node_idx_);
                cur_corner_pos += weight*node_vel*dt;
            }
            *iter = cur_corner_pos;
        }
        this->cpdi_driver_->setCurrentParticleDomain(i,particle_domain);
    }
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,2>::updateParticlePosition(Scalar dt)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    ArrayND<Vector<Scalar,2>,2> particle_domain;
    std::vector<Vector<Scalar,2> > particle_domain_vec(4);
    for(unsigned int particle_idx = 0; particle_idx < this->cpdi_driver_->particleNum(); ++particle_idx)
    {
        SolidParticle<Scalar,2> &particle = this->cpdi_driver_->particle(particle_idx);
        this->cpdi_driver_->currentParticleDomain(particle_idx,particle_domain);
        unsigned int i = 0;
        for(typename ArrayND<Vector<Scalar,2>,2>::Iterator corner_iter = particle_domain.begin(); corner_iter != particle_domain.end(); ++i,++corner_iter)
            particle_domain_vec[i] = *corner_iter;
        //coefficients
        Scalar a = (particle_domain_vec[2]-particle_domain_vec[0]).cross(particle_domain_vec[1]-particle_domain_vec[0]);
        Scalar b = (particle_domain_vec[2]-particle_domain_vec[0]).cross(particle_domain_vec[3]-particle_domain_vec[1]);
        Scalar c = (particle_domain_vec[3]-particle_domain_vec[2]).cross(particle_domain_vec[1]-particle_domain_vec[0]);
        Scalar domain_volume = a + 0.5*(b+c);
        Vector<Scalar,2> new_pos = 1.0/(24.0*domain_volume)*((6.0*domain_volume-b-c)*particle_domain_vec[0]+(6.0*domain_volume-b+c)*particle_domain_vec[1]
                                                             +(6.0*domain_volume+b-c)*particle_domain_vec[2]+(6.0*domain_volume+b+c)*particle_domain_vec[3]);
        particle.setPosition(new_pos);
    }    
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,2>::updateParticleInterpolationWeight(unsigned int particle_idx, const GridWeightFunction<Scalar,2> &weight_function,
                                                                    std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > &particle_grid_weight_and_gradient,
                                                                    unsigned int &particle_grid_pair_num,
                                                                    std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > &corner_grid_weight_and_gradient,
                                                                    std::vector<unsigned int> &corner_grid_pair_num)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    ArrayND<Vector<Scalar,2>,2> particle_domain;
    this->cpdi_driver_->currentParticleDomain(particle_idx,particle_domain);
    std::vector<Vector<Scalar,2> > particle_domain_vec;
    for(typename ArrayND<Vector<Scalar,2>,2>::Iterator corner_iter = particle_domain.begin(); corner_iter != particle_domain.end(); ++corner_iter)
        particle_domain_vec.push_back(*corner_iter);
    std::map<unsigned int,Scalar> idx_weight_map;
    std::map<unsigned int,Vector<Scalar,2> > idx_gradient_map;
    const Grid<Scalar,2> &grid = this->cpdi_driver_->grid();
    Vector<Scalar,2> grid_dx = grid.dX();
    //coefficients
    Scalar a = (particle_domain_vec[2]-particle_domain_vec[0]).cross(particle_domain_vec[1]-particle_domain_vec[0]);
    Scalar b = (particle_domain_vec[2]-particle_domain_vec[0]).cross(particle_domain_vec[3]-particle_domain_vec[1]);
    Scalar c = (particle_domain_vec[3]-particle_domain_vec[2]).cross(particle_domain_vec[1]-particle_domain_vec[0]);
    Scalar domain_volume = a + 0.5*(b+c);
    typedef UniformGridWeightFunctionInfluenceIterator<Scalar,2> InfluenceIterator;
    //first compute the weight and gradient with respect to each grid node in the influence range of the particle
    //node weight and gradient with respect to domain corners are stored as well
    for(unsigned int flat_corner_idx = 0; flat_corner_idx < particle_domain_vec.size(); ++flat_corner_idx)
    {
        corner_grid_pair_num[flat_corner_idx] = 0;
        unsigned int node_num = 0;
        for(InfluenceIterator iter(grid,particle_domain_vec[flat_corner_idx],weight_function); iter.valid(); ++node_num,++iter)
        {
            Vector<unsigned int,2> node_idx = iter.nodeIndex();
            unsigned int node_idx_1d = this->flatIndex(node_idx,grid.nodeNum());
            Vector<Scalar,2> corner_to_node = particle_domain_vec[flat_corner_idx] - grid.node(node_idx);
            for(unsigned int dim = 0; dim < 2; ++dim)
                corner_to_node[dim] /= grid_dx[dim];
            Scalar corner_weight = weight_function.weight(corner_to_node);
            //weight and gradient correspond to this node for domain corners
            corner_grid_weight_and_gradient[flat_corner_idx][node_num].node_idx_ = node_idx;
            corner_grid_weight_and_gradient[flat_corner_idx][node_num].weight_value_ = corner_weight;
            corner_grid_weight_and_gradient[flat_corner_idx][node_num].gradient_value_ = weight_function.gradient(corner_to_node);
            ++corner_grid_pair_num[flat_corner_idx];
            //weight and gradient correspond to this node for particles
            typename std::map<unsigned int,Scalar>::iterator weight_map_iter = idx_weight_map.find(node_idx_1d);
            typename std::map<unsigned int,Vector<Scalar,2> >::iterator gradient_map_iter = idx_gradient_map.find(node_idx_1d);
            switch(flat_corner_idx)
            {
            case 0:
            {
                if(weight_map_iter != idx_weight_map.end())
                    weight_map_iter->second += 1.0/(24.0*domain_volume)*(6.0*domain_volume-b-c)*corner_weight;
                else
                    idx_weight_map.insert(std::make_pair(node_idx_1d,1.0/(24.0*domain_volume)*(6.0*domain_volume-b-c)*corner_weight));
                if(gradient_map_iter != idx_gradient_map.end())
                {
                    gradient_map_iter->second[0] += 1.0/(2.0*domain_volume)*(particle_domain_vec[2][1]-particle_domain_vec[1][1])*corner_weight;
                    gradient_map_iter->second[1] += 1.0/(2.0*domain_volume)*(particle_domain_vec[1][0]-particle_domain_vec[2][0])*corner_weight;
                }
                else
                {
					Vector<Scalar,2> gradient;
                    gradient[0] = 1.0/(2.0*domain_volume)*(particle_domain_vec[2][1]-particle_domain_vec[1][1])*corner_weight;
                    gradient[1] = 1.0/(2.0*domain_volume)*(particle_domain_vec[1][0]-particle_domain_vec[2][0])*corner_weight;
					idx_gradient_map.insert(std::make_pair(node_idx_1d,gradient));
                }
                break;
            }
            case 1:
            {
                if(weight_map_iter != idx_weight_map.end())
                    weight_map_iter->second += 1.0/(24.0*domain_volume)*(6.0*domain_volume-b+c)*corner_weight;
                else
                    idx_weight_map.insert(std::make_pair(node_idx_1d,1.0/(24.0*domain_volume)*(6.0*domain_volume-b+c)*corner_weight));
                if(gradient_map_iter != idx_gradient_map.end())
                {
                    gradient_map_iter->second[0] += 1.0/(2.0*domain_volume)*(particle_domain_vec[0][1]-particle_domain_vec[3][1])*corner_weight;
                    gradient_map_iter->second[1] += 1.0/(2.0*domain_volume)*(particle_domain_vec[3][0]-particle_domain_vec[0][0])*corner_weight;
                }
                else
                {
					Vector<Scalar,2> gradient;
                    gradient[0] = 1.0/(2.0*domain_volume)*(particle_domain_vec[0][1]-particle_domain_vec[3][1])*corner_weight;
                    gradient[1] = 1.0/(2.0*domain_volume)*(particle_domain_vec[3][0]-particle_domain_vec[0][0])*corner_weight;
					idx_gradient_map.insert(std::make_pair(node_idx_1d,gradient));
                }
                break;
            }
            case 2:
            {
                if(weight_map_iter != idx_weight_map.end())
                    weight_map_iter->second += 1.0/(24.0*domain_volume)*(6.0*domain_volume+b-c)*corner_weight;
                else
                    idx_weight_map.insert(std::make_pair(node_idx_1d,1.0/(24.0*domain_volume)*(6.0*domain_volume+b-c)*corner_weight));
                if(gradient_map_iter != idx_gradient_map.end())
                {
                    gradient_map_iter->second[0] += 1.0/(2.0*domain_volume)*(particle_domain_vec[3][1]-particle_domain_vec[0][1])*corner_weight;
                    gradient_map_iter->second[1] += 1.0/(2.0*domain_volume)*(particle_domain_vec[0][0]-particle_domain_vec[3][0])*corner_weight;
                }
                else
                {
					Vector<Scalar,2> gradient;
                    gradient[0] = 1.0/(2.0*domain_volume)*(particle_domain_vec[3][1]-particle_domain_vec[0][1])*corner_weight;
                    gradient[1] = 1.0/(2.0*domain_volume)*(particle_domain_vec[0][0]-particle_domain_vec[3][0])*corner_weight;
					idx_gradient_map.insert(std::make_pair(node_idx_1d,gradient));
                }
                break;
            }
            case 3:
            {
                if(weight_map_iter != idx_weight_map.end())
                    weight_map_iter->second += 1.0/(24.0*domain_volume)*(6.0*domain_volume+b+c)*corner_weight;
                else
                    idx_weight_map.insert(std::make_pair(node_idx_1d,1.0/(24.0*domain_volume)*(6.0*domain_volume+b+c)*corner_weight));
                if(gradient_map_iter != idx_gradient_map.end())
                {
                    gradient_map_iter->second[0] += 1.0/(2.0*domain_volume)*(particle_domain_vec[1][1]-particle_domain_vec[2][1])*corner_weight;
                    gradient_map_iter->second[1] += 1.0/(2.0*domain_volume)*(particle_domain_vec[2][0]-particle_domain_vec[1][0])*corner_weight;
                }
                else
                {
					Vector<Scalar,2> gradient;
                    gradient[0] = 1.0/(2.0*domain_volume)*(particle_domain_vec[1][1]-particle_domain_vec[2][1])*corner_weight;
                    gradient[1] = 1.0/(2.0*domain_volume)*(particle_domain_vec[2][0]-particle_domain_vec[1][0])*corner_weight;
					idx_gradient_map.insert(std::make_pair(node_idx_1d,gradient));
                }
                break;
            }
            default:
                PHYSIKA_ERROR("Particle domain corner number should be 4 for Dim=2");
            }
        }
    }
    //then store the data with respect to grid nodes
    particle_grid_pair_num = 0;
    for(typename std::map<unsigned int,Scalar>::iterator iter = idx_weight_map.begin(); iter != idx_weight_map.end(); ++iter)
    {
        particle_grid_weight_and_gradient[particle_grid_pair_num].node_idx_ = this->multiDimIndex(iter->first,grid.nodeNum());
        particle_grid_weight_and_gradient[particle_grid_pair_num].weight_value_ = iter->second;
        particle_grid_weight_and_gradient[particle_grid_pair_num].gradient_value_ = idx_gradient_map[iter->first];
        ++particle_grid_pair_num;
    }
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
void CPDI2UpdateMethod<Scalar,3>::updateParticleInterpolationWeight(const GridWeightFunction<Scalar,3> &weight_function,
                                                   std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > &particle_grid_weight_and_gradient,
                                                   std::vector<unsigned int> &particle_grid_pair_num,
                                                   std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > > &corner_grid_weight_and_gradient,
                                                   std::vector<std::vector<unsigned int> > &corner_grid_pair_num)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    for(unsigned int i = 0; i < this->cpdi_driver_->particleNum(); ++i)
        updateParticleInterpolationWeight(i,weight_function,particle_grid_weight_and_gradient[i],particle_grid_pair_num[i],
                                          corner_grid_weight_and_gradient[i],corner_grid_pair_num[i]);
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,3>::updateParticleDomain(const std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > > &corner_grid_weight_and_gradient,
                                                       const std::vector<std::vector<unsigned int> > &corner_grid_pair_num, Scalar dt)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
//TO DO
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,3>::updateParticlePosition(Scalar dt)
{
//TO DO
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,3>::updateParticleInterpolationWeight(unsigned int particle_idx, const GridWeightFunction<Scalar,3> &weight_function,
                                                                    std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > &particle_grid_weight_and_gradient,
                                                                    unsigned int &particle_grid_pair_num,
                                                                    std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > &corner_grid_weight_and_gradient,
                                                                    std::vector<unsigned int> &corner_grid_pair_num)
{
//TO DO
}

//explicit instantiations
template class CPDI2UpdateMethod<float,2>;
template class CPDI2UpdateMethod<double,2>;
template class CPDI2UpdateMethod<float,3>;
template class CPDI2UpdateMethod<double,3>;

}  //end of namespace Physika
