/*
 * @file CPDI2_update_method.cpp 
 * @Brief the particle domain update procedure introduced in paper:
 *        "Second-order convected particle domain interpolation with enrichment for weak
 *         discontinuities at material interfaces"
 *    We made some key modifications(enhancements) to the conventional CPDI2 to improve
 *    its robustness with degenerated particle domain during simulation
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

#include <limits>
#include <map>
#include "Physika_Core/Arrays/array_Nd.h"
#include "Physika_Core/Grid_Weight_Functions/grid_weight_function.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Geometry/Cartesian_Grids/grid.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/Utilities/Weight_Function_Influence_Iterators/uniform_grid_weight_function_influence_iterator.h"
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
                                  std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > > &particle_grid_weight_and_gradient,
                                  std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
                                  std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,2> > > > > &corner_grid_weight,
                                  std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num,
                                  bool gradient_to_reference_coordinate)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    for(unsigned int i = 0; i < this->cpdi_driver_->objectNum(); ++i)
        for(unsigned int j = 0; j < this->cpdi_driver_->particleNumOfObject(i); ++j)
            updateParticleInterpolationWeight(i,j,weight_function,particle_grid_weight_and_gradient[i][j],particle_grid_pair_num[i][j],
                                              corner_grid_weight[i][j],corner_grid_pair_num[i][j],gradient_to_reference_coordinate);
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,2>::updateParticleInterpolationWeight(const GridWeightFunction<Scalar,2> &weight_function,
                                  const std::vector<VolumetricMesh<Scalar,2>*> &particle_domain_mesh,
                                  std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > > &particle_grid_weight_and_gradient,
                                  std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
                                  std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,2> > > > > &corner_grid_weight,
                                  std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num,
                                  bool gradient_to_reference_coordinate)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    ArrayND<Vector<Scalar,2>,2> particle_domain, initial_particle_domain;
    std::vector<Vector<Scalar,2> > particle_domain_vec(4), initial_particle_domain_vec(4);
    const Grid<Scalar,2> &grid = this->cpdi_driver_->grid();
    Vector<Scalar,2> grid_dx = grid.dX();
    Vector<unsigned int,2> grid_node_num = grid.nodeNum();
    std::map<unsigned int,Scalar> idx_weight_map;
    std::map<unsigned int,Vector<Scalar,2> > idx_gradient_map;
    std::vector<Scalar> particle_corner_weight(4);
    typedef UniformGridWeightFunctionInfluenceIterator<Scalar,2> InfluenceIterator;
    std::vector<unsigned int> global_corner_grid_pair_num;
    std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,2> > > global_corner_grid_weight;
    for(unsigned int object_idx = 0; object_idx < this->cpdi_driver_->objectNum(); ++object_idx)
    {
        //avoid redundent computation of corner grid weight
        unsigned int global_corner_num = particle_domain_mesh[object_idx]->vertNum();
        global_corner_grid_pair_num.resize(global_corner_num,0); //use 0 to indicate unintialized 
        global_corner_grid_weight.resize(global_corner_num);
        for(unsigned int particle_idx = 0; particle_idx < this->cpdi_driver_->particleNumOfObject(object_idx); ++particle_idx)
        {
            this->cpdi_driver_->currentParticleDomain(object_idx,particle_idx,particle_domain);
            this->cpdi_driver_->initialParticleDomain(object_idx,particle_idx,initial_particle_domain);
            unsigned int i = 0;
            for(typename ArrayND<Vector<Scalar,2>,2>::Iterator corner_iter = particle_domain.begin(); corner_iter != particle_domain.end(); ++corner_iter,++i)
                particle_domain_vec[i] = *corner_iter;
            i = 0;
            for(typename ArrayND<Vector<Scalar,2>,2>::Iterator corner_iter = initial_particle_domain.begin(); corner_iter != initial_particle_domain.end(); ++corner_iter,++i)
                initial_particle_domain_vec[i] = *corner_iter;
            idx_weight_map.clear();
            idx_gradient_map.clear();
            //coefficients
            Scalar a = (initial_particle_domain_vec[2]-initial_particle_domain_vec[0]).cross(initial_particle_domain_vec[1]-initial_particle_domain_vec[0]);
            Scalar b = (initial_particle_domain_vec[2]-initial_particle_domain_vec[0]).cross(initial_particle_domain_vec[3]-initial_particle_domain_vec[1]);
            Scalar c = (initial_particle_domain_vec[3]-initial_particle_domain_vec[2]).cross(initial_particle_domain_vec[1]-initial_particle_domain_vec[0]);
            Scalar domain_volume = a + 0.5*(b+c);
            //the weight between particle and domain corners
            computeParticleInterpolationWeightInParticleDomain(object_idx,particle_idx,particle_corner_weight);    
    
            //first compute the weight and gradient with respect to each grid node in the influence range of the particle
            //node weight and gradient between domain corners and grid nodes are stored as well
            for(unsigned int flat_corner_idx = 0; flat_corner_idx < particle_domain_vec.size(); ++flat_corner_idx)
            {
                Vector<unsigned int,2> multi_corner_idx = this->multiDimIndex(flat_corner_idx,Vector<unsigned int,2>(2));
                Vector<Scalar,2> gradient_integral(0);
                if(gradient_to_reference_coordinate)
                    gradient_integral = gaussIntegrateShapeFunctionGradientToReferenceCoordinateInParticleDomain(multi_corner_idx,initial_particle_domain);
                else
                    gradient_integral = gaussIntegrateShapeFunctionGradientToCurrentCoordinateInParticleDomain(multi_corner_idx,particle_domain,initial_particle_domain);
                Vector<Scalar,2> particle_corner_gradient = 1.0/domain_volume*gradient_integral;
                //now compute weight/gradient between particles and grid nodes, corners and grid nodes
                unsigned int global_corner_idx = particle_domain_mesh[object_idx]->eleVertIndex(particle_idx,flat_corner_idx);
                if(global_corner_grid_pair_num[global_corner_idx] > 0) //weight between this corner and grid nodes has been computed before
                {
                    corner_grid_pair_num[object_idx][particle_idx][flat_corner_idx] = global_corner_grid_pair_num[global_corner_idx];
                    for(unsigned int corner_grid_pair_idx = 0; corner_grid_pair_idx < global_corner_grid_pair_num[global_corner_idx]; ++corner_grid_pair_idx)
                    {
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][corner_grid_pair_idx] = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx];
                        Scalar corner_weight = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx].weight_value_;
                        Vector<unsigned int,2> node_idx = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx].node_idx_;
                        unsigned int node_idx_1d = this->flatIndex(node_idx,grid_node_num);
                        //weight and gradient correspond to this node for particles
                        typename std::map<unsigned int,Scalar>::iterator weight_map_iter = idx_weight_map.find(node_idx_1d);
                        if(weight_map_iter != idx_weight_map.end())
                        {
                            weight_map_iter->second += particle_corner_weight[flat_corner_idx]*corner_weight;
                            idx_gradient_map[node_idx_1d] += particle_corner_gradient*corner_weight;
                        }
                        else
                        {
                            idx_weight_map.insert(std::make_pair(node_idx_1d,particle_corner_weight[flat_corner_idx]*corner_weight));
                            idx_gradient_map.insert(std::make_pair(node_idx_1d,particle_corner_gradient*corner_weight));
                        }
                    }
                }
                else
                {
                    corner_grid_pair_num[object_idx][particle_idx][flat_corner_idx] = 0;
                    unsigned int node_num = 0;
                    for(InfluenceIterator iter(grid,particle_domain_vec[flat_corner_idx],weight_function); iter.valid(); ++node_num,++iter)
                    {
                        Vector<unsigned int,2> node_idx = iter.nodeIndex();
                        unsigned int node_idx_1d = this->flatIndex(node_idx,grid_node_num);
                        Vector<Scalar,2> corner_to_node = particle_domain_vec[flat_corner_idx] - grid.node(node_idx);
                        for(unsigned int dim = 0; dim < 2; ++dim)
                            corner_to_node[dim] /= grid_dx[dim];
                        Scalar corner_weight = weight_function.weight(corner_to_node);
                        //weight and gradient correspond to this node for domain corners
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][node_num].node_idx_ = node_idx;
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][node_num].weight_value_ = corner_weight;
                        ++corner_grid_pair_num[object_idx][particle_idx][flat_corner_idx];
                        //store the computed weight between domain corners and grid nodes to avoid redundent computation for other particles
                        global_corner_grid_weight[global_corner_idx].push_back(corner_grid_weight[object_idx][particle_idx][flat_corner_idx][node_num]);
                        ++global_corner_grid_pair_num[global_corner_idx];
                        //weight and gradient correspond to this node for particles
                        typename std::map<unsigned int,Scalar>::iterator weight_map_iter = idx_weight_map.find(node_idx_1d);
                        if(weight_map_iter != idx_weight_map.end())
                        {
                            weight_map_iter->second += particle_corner_weight[flat_corner_idx]*corner_weight;
                            idx_gradient_map[node_idx_1d] += particle_corner_gradient*corner_weight;
                        }
                        else
                        {
                            idx_weight_map.insert(std::make_pair(node_idx_1d,particle_corner_weight[flat_corner_idx]*corner_weight));
                            idx_gradient_map.insert(std::make_pair(node_idx_1d,particle_corner_gradient*corner_weight));
                        }
                    }
                }
            }
            //then store the data with respect to grid nodes
            particle_grid_pair_num[object_idx][particle_idx] = 0;
            for(typename std::map<unsigned int,Scalar>::iterator iter = idx_weight_map.begin(); iter != idx_weight_map.end(); ++iter)
            {
                if(iter->second > std::numeric_limits<Scalar>::epsilon()) //ignore nodes that have zero weight value, assume positive weight
                {
                    Scalar this_particle_grid_pair_num = particle_grid_pair_num[object_idx][particle_idx];
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].node_idx_ = this->multiDimIndex(iter->first,grid_node_num);
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].weight_value_ = iter->second;
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].gradient_value_ = idx_gradient_map[iter->first];
                    ++particle_grid_pair_num[object_idx][particle_idx];
                }
            }
        }
    }
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,2>::updateParticleInterpolationWeightWithEnrichment(const GridWeightFunction<Scalar,2> &weight_function,
                                  const std::vector<VolumetricMesh<Scalar,2>*> &particle_domain_mesh,
                                  const std::vector<std::vector<unsigned char> > &is_enriched_domain_corner,
                                  std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > > &particle_grid_weight_and_gradient,
                                  std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
                                  std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,2> > > > > &corner_grid_weight,
                                  std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num,
                                  bool gradient_to_reference_coordinate)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    ArrayND<Vector<Scalar,2>,2> particle_domain, initial_particle_domain;
    std::vector<Vector<Scalar,2> > particle_domain_vec(4), initial_particle_domain_vec(4);
    const Grid<Scalar,2> &grid = this->cpdi_driver_->grid();
    Vector<Scalar,2> grid_dx = grid.dX();
    Vector<unsigned int,2> grid_node_num = grid.nodeNum();
    std::map<unsigned int,Scalar> idx_weight_map;
    std::map<unsigned int,Vector<Scalar,2> > idx_gradient_map;
    std::vector<Scalar> particle_corner_weight(4);
    typedef UniformGridWeightFunctionInfluenceIterator<Scalar,2> InfluenceIterator;
    std::vector<unsigned int> global_corner_grid_pair_num;
    std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,2> > > global_corner_grid_weight;
    for(unsigned int object_idx = 0; object_idx < this->cpdi_driver_->objectNum(); ++object_idx)
    {
        //avoid redundent computation of corner grid weight
        unsigned int global_corner_num = particle_domain_mesh[object_idx]->vertNum();
        global_corner_grid_pair_num.resize(global_corner_num,0); //use 0 to indicate unintialized 
        global_corner_grid_weight.resize(global_corner_num);
        for(unsigned int particle_idx = 0; particle_idx < this->cpdi_driver_->particleNumOfObject(object_idx); ++particle_idx)
        {
            this->cpdi_driver_->currentParticleDomain(object_idx,particle_idx,particle_domain);
            this->cpdi_driver_->initialParticleDomain(object_idx,particle_idx,initial_particle_domain);
            unsigned int i = 0;
            for(typename ArrayND<Vector<Scalar,2>,2>::Iterator corner_iter = particle_domain.begin(); corner_iter != particle_domain.end(); ++corner_iter,++i)
                particle_domain_vec[i] = *corner_iter;
            i = 0;
            for(typename ArrayND<Vector<Scalar,2>,2>::Iterator corner_iter = initial_particle_domain.begin(); corner_iter != initial_particle_domain.end(); ++corner_iter,++i)
                initial_particle_domain_vec[i] = *corner_iter;
            idx_weight_map.clear();
            idx_gradient_map.clear();
            //coefficients
            Scalar a = (initial_particle_domain_vec[2]-initial_particle_domain_vec[0]).cross(initial_particle_domain_vec[1]-initial_particle_domain_vec[0]);
            Scalar b = (initial_particle_domain_vec[2]-initial_particle_domain_vec[0]).cross(initial_particle_domain_vec[3]-initial_particle_domain_vec[1]);
            Scalar c = (initial_particle_domain_vec[3]-initial_particle_domain_vec[2]).cross(initial_particle_domain_vec[1]-initial_particle_domain_vec[0]);
            Scalar domain_volume = a + 0.5*(b+c);
            //the weight between particle and domain corners
            computeParticleInterpolationWeightInParticleDomain(object_idx,particle_idx,particle_corner_weight);    
            //determine the particle type
            unsigned int enriched_corner_num = 0;
            for(unsigned int corner_idx = 0; corner_idx < 4; ++corner_idx)
            {
                unsigned int global_corner_idx = particle_domain_mesh[object_idx]->eleVertIndex(particle_idx,corner_idx);
                if(is_enriched_domain_corner[object_idx][global_corner_idx])
                    ++enriched_corner_num;
            }
    
            //first compute the weight and gradient with respect to each grid node in the influence range of the particle
            //node weight and gradient between domain corners and grid nodes are stored as well
            for(unsigned int flat_corner_idx = 0; flat_corner_idx < particle_domain_vec.size(); ++flat_corner_idx)
            {
                Vector<unsigned int,2> multi_corner_idx = this->multiDimIndex(flat_corner_idx,Vector<unsigned int,2>(2));
                Vector<Scalar,2> gradient_integral(0);
                if(gradient_to_reference_coordinate)
                    gradient_integral = gaussIntegrateShapeFunctionGradientToReferenceCoordinateInParticleDomain(multi_corner_idx,initial_particle_domain);
                else
                    gradient_integral = gaussIntegrateShapeFunctionGradientToCurrentCoordinateInParticleDomain(multi_corner_idx,particle_domain,initial_particle_domain);
                Vector<Scalar,2> particle_corner_gradient = 1.0/domain_volume*gradient_integral;
                //now compute weight/gradient between particles and grid nodes, corners and grid nodes
                unsigned int global_corner_idx = particle_domain_mesh[object_idx]->eleVertIndex(particle_idx,flat_corner_idx);
                if(global_corner_grid_pair_num[global_corner_idx] > 0) //weight between this corner and grid nodes has been computed before
                {
                    corner_grid_pair_num[object_idx][particle_idx][flat_corner_idx] = global_corner_grid_pair_num[global_corner_idx];
                    for(unsigned int corner_grid_pair_idx = 0; corner_grid_pair_idx < global_corner_grid_pair_num[global_corner_idx]; ++corner_grid_pair_idx)
                    {
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][corner_grid_pair_idx] = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx];
                        Scalar corner_weight = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx].weight_value_;
                        Vector<unsigned int,2> node_idx = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx].node_idx_;
                        unsigned int node_idx_1d = this->flatIndex(node_idx,grid_node_num);
                        //enriched domain corners do not contribute to grid
                        if(is_enriched_domain_corner[object_idx][global_corner_idx])  
                            break;
                        //weight and gradient correspond to this node for particles
                        typename std::map<unsigned int,Scalar>::iterator weight_map_iter = idx_weight_map.find(node_idx_1d);
                        if(weight_map_iter != idx_weight_map.end())
                        {
                            weight_map_iter->second += particle_corner_weight[flat_corner_idx]*corner_weight;
                            idx_gradient_map[node_idx_1d] += particle_corner_gradient*corner_weight;
                        }
                        else
                        {
                            idx_weight_map.insert(std::make_pair(node_idx_1d,particle_corner_weight[flat_corner_idx]*corner_weight));
                            idx_gradient_map.insert(std::make_pair(node_idx_1d,particle_corner_gradient*corner_weight));
                        }
                    }
                }
                else
                {
                    corner_grid_pair_num[object_idx][particle_idx][flat_corner_idx] = 0;
                    unsigned int node_num = 0;
                    for(InfluenceIterator iter(grid,particle_domain_vec[flat_corner_idx],weight_function); iter.valid(); ++node_num,++iter)
                    {
                        Vector<unsigned int,2> node_idx = iter.nodeIndex();
                        unsigned int node_idx_1d = this->flatIndex(node_idx,grid_node_num);
                        Vector<Scalar,2> corner_to_node = particle_domain_vec[flat_corner_idx] - grid.node(node_idx);
                        for(unsigned int dim = 0; dim < 2; ++dim)
                            corner_to_node[dim] /= grid_dx[dim];
                        Scalar corner_weight = weight_function.weight(corner_to_node);
                        //weight and gradient correspond to this node for domain corners
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][node_num].node_idx_ = node_idx;
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][node_num].weight_value_ = corner_weight;
                        ++corner_grid_pair_num[object_idx][particle_idx][flat_corner_idx];
                        //store the computed weight between domain corners and grid nodes to avoid redundent computation for other particles
                        global_corner_grid_weight[global_corner_idx].push_back(corner_grid_weight[object_idx][particle_idx][flat_corner_idx][node_num]);
                        ++global_corner_grid_pair_num[global_corner_idx];
                        //enriched domain corners do not contribute to grid
                        if(is_enriched_domain_corner[object_idx][global_corner_idx])  
                            break;
                        //weight and gradient correspond to this node for particles
                        typename std::map<unsigned int,Scalar>::iterator weight_map_iter = idx_weight_map.find(node_idx_1d);
                        if(weight_map_iter != idx_weight_map.end())
                        {
                            weight_map_iter->second += particle_corner_weight[flat_corner_idx]*corner_weight;
                            idx_gradient_map[node_idx_1d] += particle_corner_gradient*corner_weight;
                        }
                        else
                        {
                            idx_weight_map.insert(std::make_pair(node_idx_1d,particle_corner_weight[flat_corner_idx]*corner_weight));
                            idx_gradient_map.insert(std::make_pair(node_idx_1d,particle_corner_gradient*corner_weight));
                        }
                    }
                }
            }
            //then store the data with respect to grid nodes
            particle_grid_pair_num[object_idx][particle_idx] = 0;
            for(typename std::map<unsigned int,Scalar>::iterator iter = idx_weight_map.begin(); iter != idx_weight_map.end(); ++iter)
            {
                if(iter->second > std::numeric_limits<Scalar>::epsilon()) //ignore nodes that have zero weight value, assume positive weight
                {
                    Scalar this_particle_grid_pair_num = particle_grid_pair_num[object_idx][particle_idx];
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].node_idx_ = this->multiDimIndex(iter->first,grid_node_num);
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].weight_value_ = iter->second;
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].gradient_value_ = idx_gradient_map[iter->first];
                    ++particle_grid_pair_num[object_idx][particle_idx];
                }
            }
        }
    }
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,2>::updateParticleDomain(
    const std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,2> > > > > &corner_grid_weight,
    const std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num, Scalar dt)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    ArrayND<Vector<Scalar,2>,2> particle_domain;
    for(unsigned int obj_idx = 0; obj_idx < this->cpdi_driver_->objectNum(); ++obj_idx)
    {
        const std::vector<SolidParticle<Scalar,2>*> &particles = this->cpdi_driver_->allParticlesOfObject(obj_idx);
        for(unsigned int particle_idx = 0; particle_idx < particles.size(); ++particle_idx)
        {
            this->cpdi_driver_->currentParticleDomain(obj_idx,particle_idx,particle_domain);
            unsigned int corner_idx = 0;
            for(typename ArrayND<Vector<Scalar,2>,2>::Iterator iter = particle_domain.begin(); iter != particle_domain.end(); ++corner_idx,++iter)
            {
                Vector<Scalar,2> cur_corner_pos = *iter;
                for(unsigned int j = 0; j < corner_grid_pair_num[obj_idx][particle_idx][corner_idx]; ++j)
                {
                    Scalar weight = corner_grid_weight[obj_idx][particle_idx][corner_idx][j].weight_value_;
                    Vector<Scalar,2> node_vel = this->cpdi_driver_->gridVelocity(obj_idx,corner_grid_weight[obj_idx][particle_idx][corner_idx][j].node_idx_);
                    cur_corner_pos += weight*node_vel*dt;
                }
                *iter = cur_corner_pos;
            }
            this->cpdi_driver_->setCurrentParticleDomain(obj_idx,particle_idx,particle_domain);
        }
    }
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,2>::updateParticlePosition(Scalar dt, const std::vector<std::vector<unsigned char> > &is_dirichlet_particle)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    ArrayND<Vector<Scalar,2>,2> particle_domain;
    std::vector<Vector<Scalar,2> > particle_domain_vec(4);
    std::vector<Scalar> particle_corner_weight(4);
    for(unsigned int obj_idx = 0; obj_idx < this->cpdi_driver_->objectNum(); ++obj_idx)
    {
        for(unsigned int particle_idx = 0; particle_idx < this->cpdi_driver_->particleNumOfObject(obj_idx); ++particle_idx)
        {
            SolidParticle<Scalar,2> &particle = this->cpdi_driver_->particle(obj_idx,particle_idx);
            if(is_dirichlet_particle[obj_idx][particle_idx])  //update dirichlet particle's position with prescribed velocity
            {
                Vector<Scalar,2> new_pos = particle.position() + particle.velocity()*dt;
                particle.setPosition(new_pos);
                continue;
            }
            computeParticleInterpolationWeightInParticleDomain(obj_idx,particle_idx,particle_corner_weight);
            this->cpdi_driver_->currentParticleDomain(obj_idx,particle_idx,particle_domain);
            unsigned int i = 0;
            for(typename ArrayND<Vector<Scalar,2>,2>::Iterator iter = particle_domain.begin(); iter != particle_domain.end(); ++iter,++i)
                particle_domain_vec[i] = *iter;
            Vector<Scalar,2> new_pos(0);
            for(unsigned int flat_corner_idx = 0; flat_corner_idx < 4; ++flat_corner_idx)
                new_pos += particle_corner_weight[flat_corner_idx]*particle_domain_vec[flat_corner_idx];
            particle.setPosition(new_pos);
        }
    }    
}

template <typename Scalar>
SquareMatrix<Scalar,2> CPDI2UpdateMethod<Scalar,2>::computeParticleDeformationGradientFromDomainShape(unsigned int obj_idx, unsigned int particle_idx)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    PHYSIKA_ASSERT(obj_idx < this->cpdi_driver_->objectNum());
    PHYSIKA_ASSERT(particle_idx < this->cpdi_driver_->particleNumOfObject(obj_idx));
    ArrayND<Vector<Scalar,2>,2> initial_particle_domain;
    this->cpdi_driver_->initialParticleDomain(obj_idx,particle_idx,initial_particle_domain);
    std::vector<Vector<Scalar,2> > initial_particle_domain_vec(4);
    unsigned int i = 0;
    for(typename ArrayND<Vector<Scalar,2>,2>::Iterator iter = initial_particle_domain.begin(); iter != initial_particle_domain.end(); ++iter)
        initial_particle_domain_vec[i++] = *iter;
    //coefficients
    Scalar a = (initial_particle_domain_vec[2]-initial_particle_domain_vec[0]).cross(initial_particle_domain_vec[1]-initial_particle_domain_vec[0]);
    Scalar b = (initial_particle_domain_vec[2]-initial_particle_domain_vec[0]).cross(initial_particle_domain_vec[3]-initial_particle_domain_vec[1]);
    Scalar c = (initial_particle_domain_vec[3]-initial_particle_domain_vec[2]).cross(initial_particle_domain_vec[1]-initial_particle_domain_vec[0]);
    Scalar domain_volume = a + 0.5*(b+c);
    SquareMatrix<Scalar,2> particle_deform_grad(0);
    Vector<Scalar,2> gauss_point;
    SquareMatrix<Scalar,2> jacobian;
    //gauss quadrature in initial particle domain
    Scalar one_over_sqrt_3 = 1.0/sqrt(3.0);
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
        {
            gauss_point[0] = (2.0*i-1)*one_over_sqrt_3;
            gauss_point[1] = (2.0*j-1)*one_over_sqrt_3;
            jacobian = particleDomainJacobian(gauss_point,initial_particle_domain);
            particle_deform_grad += computeDeformationGradientAtPointInParticleDomain(obj_idx,particle_idx,gauss_point)*jacobian.determinant();
        }
    //average
    particle_deform_grad /= domain_volume;
    return particle_deform_grad;
}

template <typename Scalar>
SquareMatrix<Scalar,2> CPDI2UpdateMethod<Scalar,2>::computeDeformationGradientAtPointInParticleDomain(unsigned int obj_idx, unsigned int particle_idx,
                                                                                                      const Vector<Scalar,2> &point_natural_coordinate)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    PHYSIKA_ASSERT(obj_idx < this->cpdi_driver_->objectNum());
    PHYSIKA_ASSERT(particle_idx < this->cpdi_driver_->particleNumOfObject(obj_idx));
    PHYSIKA_ASSERT(point_natural_coordinate[0] >= -1 && point_natural_coordinate[0] <= 1);
    PHYSIKA_ASSERT(point_natural_coordinate[1] >= -1 && point_natural_coordinate[1] <= 1);
    ArrayND<Vector<Scalar,2>,2> particle_domain, initial_particle_domain;
    std::vector<Vector<Scalar,2> > particle_domain_vec(4), particle_domain_displacement(4);
    SquareMatrix<Scalar,2> identity = SquareMatrix<Scalar,2>::identityMatrix();
    this->cpdi_driver_->currentParticleDomain(obj_idx,particle_idx,particle_domain);
    this->cpdi_driver_->initialParticleDomain(obj_idx,particle_idx,initial_particle_domain);
    unsigned int i = 0;
    for(typename ArrayND<Vector<Scalar,2>,2>::Iterator corner_iter = particle_domain.begin(); corner_iter != particle_domain.end(); ++i,++corner_iter)
        particle_domain_vec[i] = *corner_iter;
    i = 0;
    for(typename ArrayND<Vector<Scalar,2>,2>::Iterator corner_iter = initial_particle_domain.begin(); corner_iter != initial_particle_domain.end(); ++i,++corner_iter)
        particle_domain_displacement[i] = particle_domain_vec[i] - (*corner_iter);
    SquareMatrix<Scalar,2> deform_grad = identity;
    i = 0;
    Vector<unsigned int,2> corner_idx;
    Vector<Scalar,2> shape_function_gradient;
    for(typename ArrayND<Vector<Scalar,2>,2>::Iterator corner_iter = particle_domain.begin(); corner_iter != particle_domain.end(); ++i,++corner_iter)
    {
        corner_idx = corner_iter.elementIndex();
        shape_function_gradient = computeShapeFunctionGradientToReferenceCoordinateAtPointInParticleDomain(obj_idx,particle_idx,corner_idx,point_natural_coordinate);
        deform_grad +=  particle_domain_displacement[i].outerProduct(shape_function_gradient);
    }
    return deform_grad;
}

template <typename Scalar>
Vector<Scalar,2> CPDI2UpdateMethod<Scalar,2>::computeShapeFunctionGradientToReferenceCoordinateAtPointInParticleDomain(unsigned int obj_idx, unsigned int particle_idx,
                                                                                          const Vector<unsigned int,2> &corner_idx, const Vector<Scalar,2> &point_natural_coordinate)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    PHYSIKA_ASSERT(obj_idx < this->cpdi_driver_->objectNum());
    PHYSIKA_ASSERT(particle_idx < this->cpdi_driver_->particleNumOfObject(obj_idx));
    PHYSIKA_ASSERT(corner_idx[0] < 2 && corner_idx[1] < 2);
    PHYSIKA_ASSERT(point_natural_coordinate[0] >= -1 && point_natural_coordinate[0] <= 1);
    PHYSIKA_ASSERT(point_natural_coordinate[1] >= -1 && point_natural_coordinate[1] <= 1);
    ArrayND<Vector<Scalar,2>,2> particle_domain, initial_particle_domain;
    this->cpdi_driver_->currentParticleDomain(obj_idx,particle_idx,particle_domain);
    this->cpdi_driver_->initialParticleDomain(obj_idx,particle_idx,initial_particle_domain);
    SquareMatrix<Scalar,2> ref_jacobian = particleDomainJacobian(point_natural_coordinate,initial_particle_domain);
    SquareMatrix<Scalar,2> ref_jacobian_inv_trans = ref_jacobian.inverse().transpose();
    Vector<Scalar,2> shape_function_derivative;
    shape_function_derivative[0] = 0.25*(2.0*corner_idx[0]-1)*(1+(2.0*corner_idx[1]-1)*point_natural_coordinate[1]);
    shape_function_derivative[1] = 0.25*(1+(2.0*corner_idx[0]-1)*point_natural_coordinate[0])*(2.0*corner_idx[1]-1);
    Vector<Scalar,2> gradient_to_ref = ref_jacobian_inv_trans*shape_function_derivative;
    return gradient_to_ref;
}

template <typename Scalar>
SquareMatrix<Scalar,2> CPDI2UpdateMethod<Scalar,2>::computeJacobianBetweenReferenceAndPrimitiveParticleDomain(unsigned int obj_idx, unsigned int particle_idx,
                                                                                                              const Vector<Scalar,2> &point_natural_coordinate)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    PHYSIKA_ASSERT(obj_idx < this->cpdi_driver_->objectNum());
    PHYSIKA_ASSERT(particle_idx < this->cpdi_driver_->particleNumOfObject(obj_idx));
    PHYSIKA_ASSERT(point_natural_coordinate[0] >= -1 && point_natural_coordinate[0] <= 1);
    PHYSIKA_ASSERT(point_natural_coordinate[1] >= -1 && point_natural_coordinate[1] <= 1);
    ArrayND<Vector<Scalar,2>,2> initial_particle_domain;
    this->cpdi_driver_->initialParticleDomain(obj_idx,particle_idx,initial_particle_domain);
    SquareMatrix<Scalar,2> ref_jacobian = particleDomainJacobian(point_natural_coordinate,initial_particle_domain);
    return ref_jacobian;
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,2>::computeParticleInterpolationWeightInParticleDomain(unsigned int obj_idx, unsigned int particle_idx, std::vector<Scalar> &particle_corner_weight)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    PHYSIKA_ASSERT(obj_idx < this->cpdi_driver_->objectNum());
    PHYSIKA_ASSERT(particle_idx < this->cpdi_driver_->particleNumOfObject(obj_idx));
    PHYSIKA_ASSERT(particle_corner_weight.size() >= 4);
    ArrayND<Vector<Scalar,2>,2> initial_particle_domain;
    this->cpdi_driver_->initialParticleDomain(obj_idx,particle_idx,initial_particle_domain);
    std::vector<Vector<Scalar,2> > initial_particle_domain_vec;
    for(typename ArrayND<Vector<Scalar,2>,2>::Iterator corner_iter = initial_particle_domain.begin(); corner_iter != initial_particle_domain.end(); ++corner_iter)
        initial_particle_domain_vec.push_back(*corner_iter);
    //coefficients
    Scalar a = (initial_particle_domain_vec[2]-initial_particle_domain_vec[0]).cross(initial_particle_domain_vec[1]-initial_particle_domain_vec[0]);
    Scalar b = (initial_particle_domain_vec[2]-initial_particle_domain_vec[0]).cross(initial_particle_domain_vec[3]-initial_particle_domain_vec[1]);
    Scalar c = (initial_particle_domain_vec[3]-initial_particle_domain_vec[2]).cross(initial_particle_domain_vec[1]-initial_particle_domain_vec[0]);
    Scalar domain_volume = a + 0.5*(b+c);
    //the weight between particle and domain corners
    particle_corner_weight[0] = 1.0/(24*domain_volume)*(6*domain_volume-b-c);
    particle_corner_weight[1] = 1.0/(24*domain_volume)*(6*domain_volume-b+c);
    particle_corner_weight[2] = 1.0/(24*domain_volume)*(6*domain_volume+b-c);
    particle_corner_weight[3] = 1.0/(24*domain_volume)*(6*domain_volume+b+c);
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,2>::computeParticleInterpolationGradientInParticleDomain(unsigned int obj_idx, unsigned int particle_idx,
                                                                                       std::vector<Vector<Scalar,2> > &particle_corner_gradient)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    PHYSIKA_ASSERT(obj_idx < this->cpdi_driver_->objectNum());
    PHYSIKA_ASSERT(particle_idx < this->cpdi_driver_->particleNumOfObject(obj_idx));
    PHYSIKA_ASSERT(particle_corner_gradient.size() >= 4);
    ArrayND<Vector<Scalar,2>,2> initial_particle_domain;
    this->cpdi_driver_->initialParticleDomain(obj_idx,particle_idx,initial_particle_domain);
    std::vector<Vector<Scalar,2> > initial_particle_domain_vec;
    for(typename ArrayND<Vector<Scalar,2>,2>::Iterator corner_iter = initial_particle_domain.begin(); corner_iter != initial_particle_domain.end(); ++corner_iter)
        initial_particle_domain_vec.push_back(*corner_iter);
    //coefficients
    Scalar a = (initial_particle_domain_vec[2]-initial_particle_domain_vec[0]).cross(initial_particle_domain_vec[1]-initial_particle_domain_vec[0]);
    Scalar b = (initial_particle_domain_vec[2]-initial_particle_domain_vec[0]).cross(initial_particle_domain_vec[3]-initial_particle_domain_vec[1]);
    Scalar c = (initial_particle_domain_vec[3]-initial_particle_domain_vec[2]).cross(initial_particle_domain_vec[1]-initial_particle_domain_vec[0]);
    Scalar domain_volume = a + 0.5*(b+c);
    //the gradient between particle and domain corners
    Vector<unsigned int,2> corner_idx(0);
    for(corner_idx[0]  = 0; corner_idx[0] < 2; ++corner_idx[0])
        for(corner_idx[1] = 0; corner_idx[1] < 2; ++corner_idx[1])
        {
            unsigned int flat_corner_idx = corner_idx[0]*2 +corner_idx[1];
            particle_corner_gradient[flat_corner_idx] = 1.0/domain_volume*gaussIntegrateShapeFunctionGradientToReferenceCoordinateInParticleDomain(corner_idx,initial_particle_domain);
        }
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,2>::updateParticleInterpolationWeight(unsigned int object_idx, unsigned int particle_idx, const GridWeightFunction<Scalar,2> &weight_function,
                                  std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > &particle_grid_weight_and_gradient,
                                  unsigned int &particle_grid_pair_num,
                                  std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,2>  > > &corner_grid_weight,
                                  std::vector<unsigned int> &corner_grid_pair_num,
                                  bool gradient_to_reference_coordinate)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    ArrayND<Vector<Scalar,2>,2> particle_domain, initial_particle_domain;
    this->cpdi_driver_->currentParticleDomain(object_idx,particle_idx,particle_domain);
    this->cpdi_driver_->initialParticleDomain(object_idx,particle_idx,initial_particle_domain);
    std::vector<Vector<Scalar,2> > particle_domain_vec(4), initial_particle_domain_vec(4);
    unsigned int i = 0;
    for(typename ArrayND<Vector<Scalar,2>,2>::Iterator corner_iter = particle_domain.begin(); corner_iter != particle_domain.end(); ++corner_iter,++i)
        particle_domain_vec[i] = *corner_iter;
    i = 0;
    for(typename ArrayND<Vector<Scalar,2>,2>::Iterator corner_iter = initial_particle_domain.begin(); corner_iter != initial_particle_domain.end(); ++corner_iter,++i)
        initial_particle_domain_vec[i] = *corner_iter;
    std::map<unsigned int,Scalar> idx_weight_map;
    std::map<unsigned int,Vector<Scalar,2> > idx_gradient_map;
    const Grid<Scalar,2> &grid = this->cpdi_driver_->grid();
    Vector<Scalar,2> grid_dx = grid.dX();
    //coefficients
    Scalar a = (initial_particle_domain_vec[2]-initial_particle_domain_vec[0]).cross(initial_particle_domain_vec[1]-initial_particle_domain_vec[0]);
    Scalar b = (initial_particle_domain_vec[2]-initial_particle_domain_vec[0]).cross(initial_particle_domain_vec[3]-initial_particle_domain_vec[1]);
    Scalar c = (initial_particle_domain_vec[3]-initial_particle_domain_vec[2]).cross(initial_particle_domain_vec[1]-initial_particle_domain_vec[0]);
    Scalar domain_volume = a + 0.5*(b+c);
    typedef UniformGridWeightFunctionInfluenceIterator<Scalar,2> InfluenceIterator;
    //first compute the weight and gradient with respect to each grid node in the influence range of the particle
    //node weight and gradient with respect to domain corners are stored as well
    for(unsigned int flat_corner_idx = 0; flat_corner_idx < particle_domain_vec.size(); ++flat_corner_idx)
    {
        corner_grid_pair_num[flat_corner_idx] = 0;
        unsigned int node_num = 0;
        Vector<unsigned int,2> corner_idx(flat_corner_idx/2,flat_corner_idx%2);
        Vector<Scalar,2> gradient_integral;
        if(gradient_to_reference_coordinate)
            gradient_integral = gaussIntegrateShapeFunctionGradientToReferenceCoordinateInParticleDomain(corner_idx, initial_particle_domain);
        else
            gradient_integral = gaussIntegrateShapeFunctionGradientToCurrentCoordinateInParticleDomain(corner_idx, particle_domain, initial_particle_domain);
        for(InfluenceIterator iter(grid,particle_domain_vec[flat_corner_idx],weight_function); iter.valid(); ++node_num,++iter)
        {
            Vector<unsigned int,2> node_idx = iter.nodeIndex();
            unsigned int node_idx_1d = this->flatIndex(node_idx,grid.nodeNum());
            Vector<Scalar,2> corner_to_node = particle_domain_vec[flat_corner_idx] - grid.node(node_idx);
            for(unsigned int dim = 0; dim < 2; ++dim)
                corner_to_node[dim] /= grid_dx[dim];
            Scalar corner_weight = weight_function.weight(corner_to_node);
            //weight correspond to this node for domain corners
            corner_grid_weight[flat_corner_idx][node_num].node_idx_ = node_idx;
            corner_grid_weight[flat_corner_idx][node_num].weight_value_ = corner_weight;
            ++corner_grid_pair_num[flat_corner_idx];
            //weight and gradient correspond to this node for particles
            typename std::map<unsigned int,Scalar>::iterator weight_map_iter = idx_weight_map.find(node_idx_1d);
            switch(flat_corner_idx)
            {
            case 0:
            {
                if(weight_map_iter != idx_weight_map.end())
                {
                    weight_map_iter->second += 1.0/(24.0*domain_volume)*(6.0*domain_volume-b-c)*corner_weight;
                    idx_gradient_map[node_idx_1d] += 1.0/domain_volume*gradient_integral*corner_weight;
                }
                else
                {
                    idx_weight_map.insert(std::make_pair(node_idx_1d,1.0/(24.0*domain_volume)*(6.0*domain_volume-b-c)*corner_weight));
					Vector<Scalar,2> gradient = 1.0/domain_volume*gradient_integral*corner_weight;
					idx_gradient_map.insert(std::make_pair(node_idx_1d,gradient));
                }
                break;
            }
            case 1:
            {
                if(weight_map_iter != idx_weight_map.end())
                {
                    weight_map_iter->second += 1.0/(24.0*domain_volume)*(6.0*domain_volume-b+c)*corner_weight;
                    idx_gradient_map[node_idx_1d] += 1.0/domain_volume*gradient_integral*corner_weight;
                }
                else
                {
                    idx_weight_map.insert(std::make_pair(node_idx_1d,1.0/(24.0*domain_volume)*(6.0*domain_volume-b+c)*corner_weight));
					Vector<Scalar,2> gradient = 1.0/domain_volume*gradient_integral*corner_weight;
					idx_gradient_map.insert(std::make_pair(node_idx_1d,gradient));
                }
                break;
            }
            case 2:
            {
                if(weight_map_iter != idx_weight_map.end())
                {
                    weight_map_iter->second += 1.0/(24.0*domain_volume)*(6.0*domain_volume+b-c)*corner_weight;
                    idx_gradient_map[node_idx_1d] += 1.0/domain_volume*gradient_integral*corner_weight;
                }
                else
                {
                    idx_weight_map.insert(std::make_pair(node_idx_1d,1.0/(24.0*domain_volume)*(6.0*domain_volume+b-c)*corner_weight));
					Vector<Scalar,2> gradient = 1.0/domain_volume*gradient_integral*corner_weight;
					idx_gradient_map.insert(std::make_pair(node_idx_1d,gradient));
                }
                break;
            }
            case 3:
            {
                if(weight_map_iter != idx_weight_map.end())
                {
                    weight_map_iter->second += 1.0/(24.0*domain_volume)*(6.0*domain_volume+b+c)*corner_weight;
                    idx_gradient_map[node_idx_1d] += 1.0/domain_volume*gradient_integral*corner_weight;
                }
                else
                {
                    idx_weight_map.insert(std::make_pair(node_idx_1d,1.0/(24.0*domain_volume)*(6.0*domain_volume+b+c)*corner_weight));
					Vector<Scalar,2> gradient = 1.0/domain_volume*gradient_integral*corner_weight;
					idx_gradient_map.insert(std::make_pair(node_idx_1d,gradient));
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
        if(iter->second > std::numeric_limits<Scalar>::epsilon()) //ignore nodes that have zero weight value, assume positive weight
        {
            particle_grid_weight_and_gradient[particle_grid_pair_num].node_idx_ = this->multiDimIndex(iter->first,grid.nodeNum());
            particle_grid_weight_and_gradient[particle_grid_pair_num].weight_value_ = iter->second;
            particle_grid_weight_and_gradient[particle_grid_pair_num].gradient_value_ = idx_gradient_map[iter->first];
            ++particle_grid_pair_num;
        }
    }
}

template <typename Scalar>
Vector<Scalar,2> CPDI2UpdateMethod<Scalar,2>::gaussIntegrateShapeFunctionGradientToCurrentCoordinateInParticleDomain(
                                              const Vector<unsigned int,2> &corner_idx, 
                                              const ArrayND<Vector<Scalar,2>,2> &particle_domain,
                                              const ArrayND<Vector<Scalar,2>,2> &initial_particle_domain)
{
    Vector<Scalar,2> result(0);
    Vector<Scalar,2> gauss_point;
    SquareMatrix<Scalar,2> jacobian, jacobian_inv_trans, ref_jacobian;
    Vector<Scalar,2> shape_function_derivative;
    //2x2 gauss integration points
    Scalar one_over_sqrt_3 = 1.0/sqrt(3.0);
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            {
                gauss_point[0] = (2.0*i-1)*one_over_sqrt_3;
                gauss_point[1] = (2.0*j-1)*one_over_sqrt_3;
                jacobian = particleDomainJacobian(gauss_point,particle_domain);
                jacobian_inv_trans = jacobian.inverse().transpose();
                ref_jacobian = particleDomainJacobian(gauss_point,initial_particle_domain);
                Scalar ref_jacobian_det = ref_jacobian.determinant();                
                shape_function_derivative[0] = 0.25*(2.0*corner_idx[0]-1)*(1+(2.0*corner_idx[1]-1)*gauss_point[1]);
                shape_function_derivative[1] = 0.25*(1+(2.0*corner_idx[0]-1)*gauss_point[0])*(2.0*corner_idx[1]-1);
                result += jacobian_inv_trans*shape_function_derivative*ref_jacobian_det;
            }
    return result;
}

template <typename Scalar>
Vector<Scalar,2> CPDI2UpdateMethod<Scalar,2>::gaussIntegrateShapeFunctionGradientToReferenceCoordinateInParticleDomain(
                                              const Vector<unsigned int,2> &corner_idx,
                                              const ArrayND<Vector<Scalar,2>,2> &initial_particle_domain)
{
    Vector<Scalar,2> result(0);
    Vector<Scalar,2> gauss_point;
    SquareMatrix<Scalar,2> ref_jacobian, ref_jacobian_inv_trans;
    Vector<Scalar,2> shape_function_derivative;
    //2x2 gauss integration points
    Scalar one_over_sqrt_3 = 1.0/sqrt(3.0);
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            {
                gauss_point[0] = (2.0*i-1)*one_over_sqrt_3;
                gauss_point[1] = (2.0*j-1)*one_over_sqrt_3;
                ref_jacobian = particleDomainJacobian(gauss_point,initial_particle_domain);
                ref_jacobian_inv_trans = ref_jacobian.inverse().transpose();
                Scalar ref_jacobian_det = ref_jacobian.determinant();                
                shape_function_derivative[0] = 0.25*(2.0*corner_idx[0]-1)*(1+(2.0*corner_idx[1]-1)*gauss_point[1]);
                shape_function_derivative[1] = 0.25*(1+(2.0*corner_idx[0]-1)*gauss_point[0])*(2.0*corner_idx[1]-1);
                result += ref_jacobian_inv_trans*shape_function_derivative*ref_jacobian_det;
            }
    return result;
}
    
template <typename Scalar>
SquareMatrix<Scalar,2> CPDI2UpdateMethod<Scalar,2>::particleDomainJacobian(const Vector<Scalar,2> &eval_point, const ArrayND<Vector<Scalar,2>,2> &particle_domain)
{
    PHYSIKA_ASSERT(eval_point[0]>=-1&&eval_point[0]<=1);
    PHYSIKA_ASSERT(eval_point[1]>=-1&&eval_point[1]<=1);
    SquareMatrix<Scalar,2> jacobian(0);
    Vector<unsigned int,2> ele_idx;
    Vector<Scalar,2> domain_corner;
    for(typename ArrayND<Vector<Scalar,2>,2>::ConstIterator iter = particle_domain.begin(); iter != particle_domain.end(); ++iter)
    {
        ele_idx = iter.elementIndex();
        domain_corner = *iter;
        for(unsigned int row = 0; row < 2; ++row)
        {
            jacobian(row,0) += 0.25*(2.0*ele_idx[0]-1)*(1+(2.0*ele_idx[1]-1)*eval_point[1])*domain_corner[row];
            jacobian(row,1) += 0.25*(1+(2.0*ele_idx[0]-1)*eval_point[0])*(2.0*ele_idx[1]-1)*domain_corner[row];
        }
    }
    return jacobian;
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
                                 std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > > &particle_grid_weight_and_gradient,
                                 std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
                                 std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,3> > > > > &corner_grid_weight,
                                 std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num,
                                 bool gradient_to_reference_coordinate)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    for(unsigned int i = 0; i < this->cpdi_driver_->objectNum(); ++i)
        for(unsigned int j = 0; j < this->cpdi_driver_->particleNumOfObject(i); ++j)
            updateParticleInterpolationWeight(i,j,weight_function,particle_grid_weight_and_gradient[i][j],particle_grid_pair_num[i][j],
                                              corner_grid_weight[i][j],corner_grid_pair_num[i][j],gradient_to_reference_coordinate);
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,3>::updateParticleInterpolationWeight(const GridWeightFunction<Scalar,3> &weight_function,
                                 const std::vector<VolumetricMesh<Scalar,3>*> &particle_domain_mesh,
                                 std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > > &particle_grid_weight_and_gradient,
                                 std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
                                 std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,3> > > > > &corner_grid_weight,
                                 std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num,
                                 bool gradient_to_reference_coordinate)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    ArrayND<Vector<Scalar,3>,3> particle_domain, initial_particle_domain;
    std::vector<Vector<Scalar,3> > particle_domain_vec(8);
    const Grid<Scalar,3> &grid = this->cpdi_driver_->grid();
    Vector<Scalar,3> grid_dx = grid.dX();
    Vector<unsigned int,3> grid_node_num = grid.nodeNum();
    std::map<unsigned int,Scalar> idx_weight_map;
    std::map<unsigned int,Vector<Scalar,3> > idx_gradient_map; 
    typedef UniformGridWeightFunctionInfluenceIterator<Scalar,3> InfluenceIterator;
    std::vector<unsigned int> global_corner_grid_pair_num;
    std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,3> > > global_corner_grid_weight;
    for(unsigned int object_idx = 0; object_idx < this->cpdi_driver_->objectNum(); ++object_idx)
    {
        //avoid redundent computation of corner grid weight
        unsigned int global_corner_num = particle_domain_mesh[object_idx]->vertNum();
        global_corner_grid_pair_num.resize(global_corner_num,0); //use 0 to indicate unintialized 
        global_corner_grid_weight.resize(global_corner_num);
        for(unsigned int particle_idx = 0; particle_idx < this->cpdi_driver_->particleNumOfObject(object_idx); ++particle_idx)
        {
            this->cpdi_driver_->currentParticleDomain(object_idx,particle_idx,particle_domain);
            this->cpdi_driver_->initialParticleDomain(object_idx,particle_idx,initial_particle_domain);
            unsigned int flat_corner_idx = 0;
            for(typename ArrayND<Vector<Scalar,3>,3>::Iterator corner_iter = particle_domain.begin(); corner_iter != particle_domain.end(); ++corner_iter)
                particle_domain_vec[flat_corner_idx++] = (*corner_iter);
            idx_weight_map.clear();
            idx_gradient_map.clear();
            Scalar domain_volume = particleDomainVolume(initial_particle_domain);

            //first compute the weight and gradient with respect to each grid node in the influence range of the particle
            //node weight and gradient with respect to domain corners are stored as well
            for(flat_corner_idx = 0; flat_corner_idx < particle_domain_vec.size(); ++flat_corner_idx)
            {
                Vector<unsigned int,3> multi_corner_idx = this->multiDimIndex(flat_corner_idx,Vector<unsigned int,3>(2));
                Scalar approximate_integrate_shape_function_in_domain = gaussIntegrateShapeFunctionValueInParticleDomain(multi_corner_idx,initial_particle_domain);
                Vector<Scalar,3> approximate_integrate_shape_function_gradient_in_domain(0);
                if(gradient_to_reference_coordinate)
                {                            
                    approximate_integrate_shape_function_gradient_in_domain =
                        gaussIntegrateShapeFunctionGradientToReferenceCoordinateInParticleDomain(multi_corner_idx,initial_particle_domain);
                }
                else
                {                            
                    approximate_integrate_shape_function_gradient_in_domain =
                        gaussIntegrateShapeFunctionGradientToCurrentCoordinateInParticleDomain(multi_corner_idx,particle_domain,initial_particle_domain);
                }
                //weight and gradient between particle and domain corners
                Scalar particle_corner_weight = 1.0/domain_volume*approximate_integrate_shape_function_in_domain;
                Vector<Scalar,3> particle_corner_gradient = 1.0/domain_volume*approximate_integrate_shape_function_gradient_in_domain;
                //now compute weight/gradient between particles and grid nodes, corners and grid nodes
                unsigned int global_corner_idx = particle_domain_mesh[object_idx]->eleVertIndex(particle_idx,flat_corner_idx);
                if(global_corner_grid_pair_num[global_corner_idx] > 0) //weight between this corner and grid nodes has been computed before
                {
                    corner_grid_pair_num[object_idx][particle_idx][flat_corner_idx] = global_corner_grid_pair_num[global_corner_idx];
                    for(unsigned int corner_grid_pair_idx = 0; corner_grid_pair_idx < global_corner_grid_pair_num[global_corner_idx]; ++corner_grid_pair_idx)
                    {
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][corner_grid_pair_idx] = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx];
                        Scalar corner_weight = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx].weight_value_;
                        Vector<unsigned int,3> node_idx = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx].node_idx_;
                        unsigned int node_idx_1d = this->flatIndex(node_idx,grid_node_num);
                        //weight and gradient correspond to this node for particles
                        typename std::map<unsigned int,Scalar>::iterator weight_map_iter = idx_weight_map.find(node_idx_1d);
                        if(weight_map_iter != idx_weight_map.end())
                        {
                            weight_map_iter->second += particle_corner_weight*corner_weight;
                            idx_gradient_map[node_idx_1d] += particle_corner_gradient*corner_weight;
                        }
                        else
                        {
                            idx_weight_map.insert(std::make_pair(node_idx_1d,particle_corner_weight*corner_weight));
                            idx_gradient_map.insert(std::make_pair(node_idx_1d,particle_corner_gradient*corner_weight));
                        }
                    }
                }
                else
                {
                    corner_grid_pair_num[object_idx][particle_idx][flat_corner_idx] = 0;
                    unsigned int node_num = 0;
                    for(InfluenceIterator iter(grid,particle_domain_vec[flat_corner_idx],weight_function); iter.valid(); ++node_num,++iter)
                    {
                        Vector<unsigned int,3> node_idx = iter.nodeIndex();
                        unsigned int node_idx_1d = this->flatIndex(node_idx,grid_node_num);
                        Vector<Scalar,3> corner_to_node = particle_domain_vec[flat_corner_idx] - grid.node(node_idx);
                        for(unsigned int dim = 0; dim < 3; ++dim)
                            corner_to_node[dim] /= grid_dx[dim];
                        Scalar corner_weight = weight_function.weight(corner_to_node);
                        //weight correspond to this node for domain corners
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][node_num].node_idx_ = node_idx;
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][node_num].weight_value_ = corner_weight;
                        ++corner_grid_pair_num[object_idx][particle_idx][flat_corner_idx];
                        //store the computed weight between domain corners and grid nodes to avoid redundent computation for other particles
                        global_corner_grid_weight[global_corner_idx].push_back(corner_grid_weight[object_idx][particle_idx][flat_corner_idx][node_num]);
                        ++global_corner_grid_pair_num[global_corner_idx];
                        //weight and gradient correspond to this node for particles
                        typename std::map<unsigned int,Scalar>::iterator weight_map_iter = idx_weight_map.find(node_idx_1d);
                        if(weight_map_iter != idx_weight_map.end())
                        {
                            weight_map_iter->second += particle_corner_weight*corner_weight;
                            idx_gradient_map[node_idx_1d] += particle_corner_gradient*corner_weight;
                        }
                        else
                        {
                            idx_weight_map.insert(std::make_pair(node_idx_1d,particle_corner_weight*corner_weight));
                            idx_gradient_map.insert(std::make_pair(node_idx_1d,particle_corner_gradient*corner_weight));
                        }
                    }
                }
            }
            //then store the data with respect to grid nodes
            particle_grid_pair_num[object_idx][particle_idx] = 0;
            for(typename std::map<unsigned int,Scalar>::iterator iter = idx_weight_map.begin(); iter != idx_weight_map.end(); ++iter)
            {
                if(iter->second > std::numeric_limits<Scalar>::epsilon()) //ignore nodes that have zero weight value, assume positive weight
                {
                    Scalar this_particle_grid_pair_num = particle_grid_pair_num[object_idx][particle_idx];
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].node_idx_ = this->multiDimIndex(iter->first,grid_node_num);
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].weight_value_ = iter->second;
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].gradient_value_ = idx_gradient_map[iter->first];
                    ++particle_grid_pair_num[object_idx][particle_idx];
                }
            }
        }
    }
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,3>::updateParticleInterpolationWeightWithEnrichment(const GridWeightFunction<Scalar,3> &weight_function,
                                  const std::vector<VolumetricMesh<Scalar,3>*> &particle_domain_mesh,
                                  const std::vector<std::vector<unsigned char> > &is_enriched_domain_corner,
                                  std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > > &particle_grid_weight_and_gradient,
                                  std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
                                  std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,3>  > > > > &corner_grid_weight,
                                  std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num,
                                  bool gradient_to_reference_coordinate)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    ArrayND<Vector<Scalar,3>,3> particle_domain, initial_particle_domain;
    std::vector<Vector<Scalar,3> > particle_domain_vec(8);
    const Grid<Scalar,3> &grid = this->cpdi_driver_->grid();
    Vector<Scalar,3> grid_dx = grid.dX();
    Vector<unsigned int,3> grid_node_num = grid.nodeNum();
    std::map<unsigned int,Scalar> idx_weight_map;
    std::map<unsigned int,Vector<Scalar,3> > idx_gradient_map; 
    typedef UniformGridWeightFunctionInfluenceIterator<Scalar,3> InfluenceIterator;
    std::vector<unsigned int> global_corner_grid_pair_num;
    std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,3> > > global_corner_grid_weight;
    for(unsigned int object_idx = 0; object_idx < this->cpdi_driver_->objectNum(); ++object_idx)
    {
        //avoid redundent computation of corner grid weight
        unsigned int global_corner_num = particle_domain_mesh[object_idx]->vertNum();
        global_corner_grid_pair_num.resize(global_corner_num,0); //use 0 to indicate unintialized 
        global_corner_grid_weight.resize(global_corner_num);
        for(unsigned int particle_idx = 0; particle_idx < this->cpdi_driver_->particleNumOfObject(object_idx); ++particle_idx)
        {
            this->cpdi_driver_->currentParticleDomain(object_idx,particle_idx,particle_domain);
            this->cpdi_driver_->initialParticleDomain(object_idx,particle_idx,initial_particle_domain);
            unsigned int flat_corner_idx = 0;
            for(typename ArrayND<Vector<Scalar,3>,3>::Iterator corner_iter = particle_domain.begin(); corner_iter != particle_domain.end(); ++corner_iter)
                particle_domain_vec[flat_corner_idx++] = (*corner_iter);
            idx_weight_map.clear();
            idx_gradient_map.clear();
            Scalar domain_volume = particleDomainVolume(initial_particle_domain);
            //determine the particle type
            unsigned int enriched_corner_num = 0;
            for(unsigned int corner_idx = 0; corner_idx < 8; ++corner_idx)
            {
                unsigned int global_corner_idx = particle_domain_mesh[object_idx]->eleVertIndex(particle_idx,corner_idx);
                if(is_enriched_domain_corner[object_idx][global_corner_idx])
                    ++enriched_corner_num;
            }

            //first compute the weight and gradient with respect to each grid node in the influence range of the particle
            //node weight and gradient with respect to domain corners are stored as well
            for(flat_corner_idx = 0; flat_corner_idx < particle_domain_vec.size(); ++flat_corner_idx)
            {
                Vector<unsigned int,3> multi_corner_idx = this->multiDimIndex(flat_corner_idx,Vector<unsigned int,3>(2));
                Scalar approximate_integrate_shape_function_in_domain = gaussIntegrateShapeFunctionValueInParticleDomain(multi_corner_idx,initial_particle_domain);
                Vector<Scalar,3> approximate_integrate_shape_function_gradient_in_domain(0);
                if(gradient_to_reference_coordinate)
                {                            
                    approximate_integrate_shape_function_gradient_in_domain =
                        gaussIntegrateShapeFunctionGradientToReferenceCoordinateInParticleDomain(multi_corner_idx,initial_particle_domain);
                }
                else
                {                            
                    approximate_integrate_shape_function_gradient_in_domain =
                        gaussIntegrateShapeFunctionGradientToCurrentCoordinateInParticleDomain(multi_corner_idx,particle_domain,initial_particle_domain);
                }
                //weight and gradient between particle and domain corners
                Scalar particle_corner_weight = 1.0/domain_volume*approximate_integrate_shape_function_in_domain;
                Vector<Scalar,3> particle_corner_gradient = 1.0/domain_volume*approximate_integrate_shape_function_gradient_in_domain;
                //now compute weight/gradient between particles and grid nodes, corners and grid nodes
                unsigned int global_corner_idx = particle_domain_mesh[object_idx]->eleVertIndex(particle_idx,flat_corner_idx);
                if(global_corner_grid_pair_num[global_corner_idx] > 0) //weight between this corner and grid nodes has been computed before
                {
                    corner_grid_pair_num[object_idx][particle_idx][flat_corner_idx] = global_corner_grid_pair_num[global_corner_idx];
                    for(unsigned int corner_grid_pair_idx = 0; corner_grid_pair_idx < global_corner_grid_pair_num[global_corner_idx]; ++corner_grid_pair_idx)
                    {
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][corner_grid_pair_idx] = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx];
                        Scalar corner_weight = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx].weight_value_;
                        Vector<unsigned int,3> node_idx = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx].node_idx_;
                        unsigned int node_idx_1d = this->flatIndex(node_idx,grid_node_num);
                        //enriched domain corners do not contribute to grid
                        if(is_enriched_domain_corner[object_idx][global_corner_idx])  
                            break;
                        //weight and gradient correspond to this node for particles
                        typename std::map<unsigned int,Scalar>::iterator weight_map_iter = idx_weight_map.find(node_idx_1d);
                        if(weight_map_iter != idx_weight_map.end())
                        {
                            weight_map_iter->second += particle_corner_weight*corner_weight;
                            idx_gradient_map[node_idx_1d] += particle_corner_gradient*corner_weight;
                        }
                        else
                        {
                            idx_weight_map.insert(std::make_pair(node_idx_1d,particle_corner_weight*corner_weight));
                            idx_gradient_map.insert(std::make_pair(node_idx_1d,particle_corner_gradient*corner_weight));
                        }
                    }
                }
                else
                {
                    corner_grid_pair_num[object_idx][particle_idx][flat_corner_idx] = 0;
                    unsigned int node_num = 0;
                    for(InfluenceIterator iter(grid,particle_domain_vec[flat_corner_idx],weight_function); iter.valid(); ++node_num,++iter)
                    {
                        Vector<unsigned int,3> node_idx = iter.nodeIndex();
                        unsigned int node_idx_1d = this->flatIndex(node_idx,grid_node_num);
                        Vector<Scalar,3> corner_to_node = particle_domain_vec[flat_corner_idx] - grid.node(node_idx);
                        for(unsigned int dim = 0; dim < 3; ++dim)
                            corner_to_node[dim] /= grid_dx[dim];
                        Scalar corner_weight = weight_function.weight(corner_to_node);
                        //weight correspond to this node for domain corners
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][node_num].node_idx_ = node_idx;
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][node_num].weight_value_ = corner_weight;
                        ++corner_grid_pair_num[object_idx][particle_idx][flat_corner_idx];
                        //store the computed weight between domain corners and grid nodes to avoid redundent computation for other particles
                        global_corner_grid_weight[global_corner_idx].push_back(corner_grid_weight[object_idx][particle_idx][flat_corner_idx][node_num]);
                        ++global_corner_grid_pair_num[global_corner_idx];
                        //enriched domain corners do not contribute to grid
                        if(is_enriched_domain_corner[object_idx][global_corner_idx])  
                            break;
                        //weight and gradient correspond to this node for particles
                        typename std::map<unsigned int,Scalar>::iterator weight_map_iter = idx_weight_map.find(node_idx_1d);
                        if(weight_map_iter != idx_weight_map.end())
                        {
                            weight_map_iter->second += particle_corner_weight*corner_weight;
                            idx_gradient_map[node_idx_1d] += particle_corner_gradient*corner_weight;
                        }
                        else
                        {
                            idx_weight_map.insert(std::make_pair(node_idx_1d,particle_corner_weight*corner_weight));
                            idx_gradient_map.insert(std::make_pair(node_idx_1d,particle_corner_gradient*corner_weight));
                        }
                    }
                }
            }
            //then store the data with respect to grid nodes
            particle_grid_pair_num[object_idx][particle_idx] = 0;
            for(typename std::map<unsigned int,Scalar>::iterator iter = idx_weight_map.begin(); iter != idx_weight_map.end(); ++iter)
            {
                if(iter->second > std::numeric_limits<Scalar>::epsilon()) //ignore nodes that have zero weight value, assume positive weight
                {
                    Scalar this_particle_grid_pair_num = particle_grid_pair_num[object_idx][particle_idx];
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].node_idx_ = this->multiDimIndex(iter->first,grid_node_num);
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].weight_value_ = iter->second;
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].gradient_value_ = idx_gradient_map[iter->first];
                    ++particle_grid_pair_num[object_idx][particle_idx];
                }
            }
        }
    }
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,3>::updateParticleDomain(
    const std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,3> > > > > &corner_grid_weight,
    const std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num, Scalar dt)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    ArrayND<Vector<Scalar,3>,3> particle_domain;
    for(unsigned int obj_idx = 0; obj_idx < this->cpdi_driver_->objectNum(); ++obj_idx)
    {
        for(unsigned int particle_idx = 0; particle_idx < this->cpdi_driver_->particleNumOfObject(obj_idx); ++particle_idx)
        {
            this->cpdi_driver_->currentParticleDomain(obj_idx,particle_idx,particle_domain);
            unsigned int corner_idx = 0;
            for(typename ArrayND<Vector<Scalar,3>,3>::Iterator iter = particle_domain.begin(); iter != particle_domain.end(); ++corner_idx,++iter)
            {
                Vector<Scalar,3> cur_corner_pos = *iter;
                for(unsigned int j = 0; j < corner_grid_pair_num[obj_idx][particle_idx][corner_idx]; ++j)
                {
                    Scalar weight = corner_grid_weight[obj_idx][particle_idx][corner_idx][j].weight_value_;
                    Vector<Scalar,3> node_vel = this->cpdi_driver_->gridVelocity(obj_idx,corner_grid_weight[obj_idx][particle_idx][corner_idx][j].node_idx_);
                    cur_corner_pos += weight*node_vel*dt;
                }
                *iter = cur_corner_pos;
            }
            this->cpdi_driver_->setCurrentParticleDomain(obj_idx,particle_idx,particle_domain);
        }
    }
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,3>::updateParticlePosition(Scalar dt, const std::vector<std::vector<unsigned char> > &is_dirichlet_particle)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    ArrayND<Vector<Scalar,3>,3> particle_domain;
    std::vector<Vector<Scalar,3> > particle_domain_vec(8);
    std::vector<Scalar> particle_corner_weight(8);
    for(unsigned int obj_idx = 0; obj_idx < this->cpdi_driver_->objectNum(); ++obj_idx)
    {
        for(unsigned int particle_idx = 0; particle_idx < this->cpdi_driver_->particleNumOfObject(obj_idx); ++particle_idx)
        {
            SolidParticle<Scalar,3> &particle = this->cpdi_driver_->particle(obj_idx,particle_idx);
            if(is_dirichlet_particle[obj_idx][particle_idx])  //update dirichlet particle's position with prescribed velocity
            {
                Vector<Scalar,3> new_pos = particle.position() + particle.velocity()*dt;
                particle.setPosition(new_pos);
                continue;
            }
            this->cpdi_driver_->currentParticleDomain(obj_idx,particle_idx,particle_domain);
            unsigned int i = 0;
            for(typename ArrayND<Vector<Scalar,3>,3>::Iterator corner_iter = particle_domain.begin(); corner_iter != particle_domain.end(); ++i, ++corner_iter)
                particle_domain_vec[i] = *corner_iter;
            computeParticleInterpolationWeightInParticleDomain(obj_idx,particle_idx,particle_corner_weight);
            Vector<Scalar,3> new_pos(0);
            for(unsigned int flat_corner_idx = 0; flat_corner_idx < 8; ++flat_corner_idx)
                new_pos += particle_corner_weight[flat_corner_idx]*particle_domain_vec[flat_corner_idx];
            particle.setPosition(new_pos);
        }
    }
}

template <typename Scalar>
SquareMatrix<Scalar,3> CPDI2UpdateMethod<Scalar,3>::computeParticleDeformationGradientFromDomainShape(unsigned int obj_idx, unsigned int particle_idx)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    PHYSIKA_ASSERT(obj_idx < this->cpdi_driver_->objectNum());
    PHYSIKA_ASSERT(particle_idx < this->cpdi_driver_->particleNumOfObject(obj_idx));
    ArrayND<Vector<Scalar,3>,3> initial_particle_domain;
    this->cpdi_driver_->initialParticleDomain(obj_idx,particle_idx,initial_particle_domain);
    SquareMatrix<Scalar,3> particle_deform_grad(0);
    //gauss quadrature in initial particle domain
    Scalar one_over_sqrt_3 = 1.0/sqrt(3.0);
    Vector<Scalar,3> gauss_point;
    SquareMatrix<Scalar,3> jacobian;
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            for(unsigned int k = 0; k < 2; ++k)
            {
                gauss_point[0] = (2.0*i-1)*one_over_sqrt_3;
                gauss_point[1] = (2.0*j-1)*one_over_sqrt_3;
                gauss_point[2] = (2.0*k-1)*one_over_sqrt_3;
                jacobian = particleDomainJacobian(gauss_point,initial_particle_domain);
                particle_deform_grad += computeDeformationGradientAtPointInParticleDomain(obj_idx,particle_idx,gauss_point)*jacobian.determinant();
            }
    Scalar particle_domain_volume = particleDomainVolume(initial_particle_domain);
    PHYSIKA_ASSERT(particle_domain_volume > 0);
    return particle_deform_grad/particle_domain_volume;
}

template <typename Scalar>
SquareMatrix<Scalar,3> CPDI2UpdateMethod<Scalar,3>::computeDeformationGradientAtPointInParticleDomain(unsigned int obj_idx, unsigned int particle_idx,
                                                                                                      const Vector<Scalar,3> &point_natural_coordinate)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    PHYSIKA_ASSERT(obj_idx < this->cpdi_driver_->objectNum());
    PHYSIKA_ASSERT(particle_idx < this->cpdi_driver_->particleNumOfObject(obj_idx));
    PHYSIKA_ASSERT(point_natural_coordinate[0] >= -1 && point_natural_coordinate[0] <= 1);
    PHYSIKA_ASSERT(point_natural_coordinate[1] >= -1 && point_natural_coordinate[1] <= 1);
    PHYSIKA_ASSERT(point_natural_coordinate[2] >= -1 && point_natural_coordinate[2] <= 1);
    ArrayND<Vector<Scalar,3>,3> particle_domain, initial_particle_domain;
    std::vector<Vector<Scalar,3> > particle_domain_vec(8), particle_domain_displacement(8);
    SquareMatrix<Scalar,3> identity = SquareMatrix<Scalar,3>::identityMatrix();
    this->cpdi_driver_->currentParticleDomain(obj_idx,particle_idx,particle_domain);
    this->cpdi_driver_->initialParticleDomain(obj_idx,particle_idx,initial_particle_domain);
    unsigned int i = 0;
    for(typename ArrayND<Vector<Scalar,3>,3>::Iterator corner_iter = particle_domain.begin(); corner_iter != particle_domain.end(); ++i,++corner_iter)
        particle_domain_vec[i] = *corner_iter;
    i = 0;
    for(typename ArrayND<Vector<Scalar,3>,3>::Iterator corner_iter = initial_particle_domain.begin(); corner_iter != initial_particle_domain.end(); ++i,++corner_iter)
        particle_domain_displacement[i] = particle_domain_vec[i] - (*corner_iter);
    SquareMatrix<Scalar,3> deform_grad = identity;
    i = 0;
    for(typename ArrayND<Vector<Scalar,3>,3>::Iterator corner_iter = particle_domain.begin(); corner_iter != particle_domain.end(); ++i,++corner_iter)
    {
        Vector<unsigned int,3> corner_idx = corner_iter.elementIndex();
        Vector<Scalar,3> shape_function_gradient = computeShapeFunctionGradientToReferenceCoordinateAtPointInParticleDomain(obj_idx,particle_idx,corner_idx,point_natural_coordinate);
        deform_grad +=  particle_domain_displacement[i].outerProduct(shape_function_gradient);
    }
    return deform_grad;
}

template <typename Scalar>
Vector<Scalar,3> CPDI2UpdateMethod<Scalar,3>::computeShapeFunctionGradientToReferenceCoordinateAtPointInParticleDomain(unsigned int obj_idx, unsigned int particle_idx,
                                                                                                                       const Vector<unsigned int,3> &corner_idx, const Vector<Scalar,3> &point_natural_coordinate)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    PHYSIKA_ASSERT(obj_idx < this->cpdi_driver_->objectNum());
    PHYSIKA_ASSERT(particle_idx < this->cpdi_driver_->particleNumOfObject(obj_idx));
    PHYSIKA_ASSERT(corner_idx[0] < 2 && corner_idx[1] < 2 && corner_idx[2] < 2);
    PHYSIKA_ASSERT(point_natural_coordinate[0] >= -1 && point_natural_coordinate[0] <= 1);
    PHYSIKA_ASSERT(point_natural_coordinate[1] >= -1 && point_natural_coordinate[1] <= 1);
    PHYSIKA_ASSERT(point_natural_coordinate[2] >= -1 && point_natural_coordinate[2] <= 1);
    ArrayND<Vector<Scalar,3>,3> initial_particle_domain;
    this->cpdi_driver_->initialParticleDomain(obj_idx,particle_idx,initial_particle_domain);
    SquareMatrix<Scalar,3> ref_jacobian = particleDomainJacobian(point_natural_coordinate,initial_particle_domain);
    SquareMatrix<Scalar,3> ref_jacobian_inv_trans = ref_jacobian.inverse().transpose();
    Vector<Scalar,3> shape_function_derivative;
    shape_function_derivative[0] = 0.125*(2.0*corner_idx[0]-1)*(1+(2.0*corner_idx[1]-1)*point_natural_coordinate[1])*(1+(2.0*corner_idx[2]-1)*point_natural_coordinate[2]);
    shape_function_derivative[1] = 0.125*(1+(2.0*corner_idx[0]-1)*point_natural_coordinate[0])*(2.0*corner_idx[1]-1)*(1+(2.0*corner_idx[2]-1)*point_natural_coordinate[2]);
    shape_function_derivative[2] = 0.125*(1+(2.0*corner_idx[0]-1)*point_natural_coordinate[0])*(1+(2.0*corner_idx[1]-1)*point_natural_coordinate[1])*(2.0*corner_idx[2]-1);
    Vector<Scalar,3> gradient_to_ref = ref_jacobian_inv_trans*shape_function_derivative;
    return gradient_to_ref;
}

template <typename Scalar>
SquareMatrix<Scalar,3> CPDI2UpdateMethod<Scalar,3>::computeJacobianBetweenReferenceAndPrimitiveParticleDomain(unsigned int obj_idx, unsigned int particle_idx,
                                                                                                              const Vector<Scalar,3> &point_natural_coordinate)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    PHYSIKA_ASSERT(obj_idx < this->cpdi_driver_->objectNum());
    PHYSIKA_ASSERT(particle_idx < this->cpdi_driver_->particleNumOfObject(obj_idx));
    PHYSIKA_ASSERT(point_natural_coordinate[0] >= -1 && point_natural_coordinate[0] <= 1);
    PHYSIKA_ASSERT(point_natural_coordinate[1] >= -1 && point_natural_coordinate[1] <= 1);
    PHYSIKA_ASSERT(point_natural_coordinate[2] >= -1 && point_natural_coordinate[2] <= 1);
    ArrayND<Vector<Scalar,3>,3> initial_particle_domain;
    this->cpdi_driver_->initialParticleDomain(obj_idx,particle_idx,initial_particle_domain);
    SquareMatrix<Scalar,3> ref_jacobian = particleDomainJacobian(point_natural_coordinate,initial_particle_domain);
    return ref_jacobian;
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,3>::computeParticleInterpolationWeightInParticleDomain(unsigned int obj_idx, unsigned int particle_idx, std::vector<Scalar> &particle_corner_weight)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    PHYSIKA_ASSERT(obj_idx < this->cpdi_driver_->objectNum());
    PHYSIKA_ASSERT(particle_idx < this->cpdi_driver_->particleNumOfObject(obj_idx));
    PHYSIKA_ASSERT(particle_corner_weight.size() >= 8);
    ArrayND<Vector<Scalar,3>,3> initial_particle_domain;
    this->cpdi_driver_->initialParticleDomain(obj_idx,particle_idx,initial_particle_domain);
    Scalar particle_domain_volume = particleDomainVolume(initial_particle_domain);
    Vector<unsigned int,3> corner_idx(0);
    //eight corners
    unsigned int corner_idx_1d = 0;
    for(corner_idx[0] = 0; corner_idx[0] < 2; ++corner_idx[0])
        for(corner_idx[1] = 0; corner_idx[1] < 2; ++corner_idx[1])
            for(corner_idx[2] = 0; corner_idx[2] < 2; ++corner_idx[2])
            {
                //integrate
                particle_corner_weight[corner_idx_1d] = gaussIntegrateShapeFunctionValueInParticleDomain(corner_idx,initial_particle_domain);
                //average
                particle_corner_weight[corner_idx_1d] /= particle_domain_volume;
                ++corner_idx_1d;
            }
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,3>::computeParticleInterpolationGradientInParticleDomain(unsigned int obj_idx, unsigned int particle_idx,
                                                                                                                                              std::vector<Vector<Scalar,3> > &particle_corner_gradient)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    PHYSIKA_ASSERT(obj_idx < this->cpdi_driver_->objectNum());
    PHYSIKA_ASSERT(particle_idx < this->cpdi_driver_->particleNumOfObject(obj_idx));
    PHYSIKA_ASSERT(particle_corner_gradient.size() >= 8);
    ArrayND<Vector<Scalar,3>,3> initial_particle_domain;
    this->cpdi_driver_->initialParticleDomain(obj_idx,particle_idx,initial_particle_domain);
    Scalar particle_domain_volume = particleDomainVolume(initial_particle_domain);
    Vector<unsigned int,3> corner_idx(0);
    //eight corners
    unsigned int corner_idx_1d = 0;
    for(corner_idx[0] = 0; corner_idx[0] < 2; ++corner_idx[0])
        for(corner_idx[1] = 0; corner_idx[1] < 2; ++corner_idx[1])
            for(corner_idx[2] = 0; corner_idx[2] < 2; ++corner_idx[2])
            {
                //integrate
                particle_corner_gradient[corner_idx_1d] = gaussIntegrateShapeFunctionGradientToReferenceCoordinateInParticleDomain(corner_idx,initial_particle_domain);
                //average
                particle_corner_gradient[corner_idx_1d] /= particle_domain_volume;
                ++corner_idx_1d;
            }
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,3>::updateParticleInterpolationWeight(unsigned int object_idx, unsigned int particle_idx, const GridWeightFunction<Scalar,3> &weight_function,
                                                                    std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > &particle_grid_weight_and_gradient,
                                                                    unsigned int &particle_grid_pair_num,
                                                                    std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,3> > > &corner_grid_weight,
                                                                    std::vector<unsigned int> &corner_grid_pair_num,
                                                                    bool gradient_to_reference_coordinate)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    ArrayND<Vector<Scalar,3>,3> particle_domain,initial_particle_domain;
    this->cpdi_driver_->currentParticleDomain(object_idx,particle_idx,particle_domain);
    this->cpdi_driver_->initialParticleDomain(object_idx,particle_idx,initial_particle_domain);
    std::vector<Vector<Scalar,3> > particle_domain_vec;
    for(typename ArrayND<Vector<Scalar,3>,3>::Iterator corner_iter = particle_domain.begin(); corner_iter != particle_domain.end(); ++corner_iter)
        particle_domain_vec.push_back(*corner_iter);
    std::map<unsigned int,Scalar> idx_weight_map;
    std::map<unsigned int,Vector<Scalar,3> > idx_gradient_map;
    const Grid<Scalar,3> &grid = this->cpdi_driver_->grid();
    Vector<Scalar,3> grid_dx = grid.dX();
    Scalar domain_volume = particleDomainVolume(initial_particle_domain);

    typedef UniformGridWeightFunctionInfluenceIterator<Scalar,3> InfluenceIterator;
    //first compute the weight and gradient with respect to each grid node in the influence range of the particle
    //node weight and gradient with respect to domain corners are stored as well
    for(unsigned int flat_corner_idx = 0; flat_corner_idx < particle_domain_vec.size(); ++flat_corner_idx)
    {
        corner_grid_pair_num[flat_corner_idx] = 0;
        unsigned int node_num = 0;
        Vector<unsigned int,3> multi_corner_idx = this->multiDimIndex(flat_corner_idx,Vector<unsigned int,3>(2));
        Scalar approximate_integrate_shape_function_in_domain = gaussIntegrateShapeFunctionValueInParticleDomain(multi_corner_idx,initial_particle_domain);
        Vector<Scalar,3> approximate_integrate_shape_function_gradient_in_domain;
        if(gradient_to_reference_coordinate)
            approximate_integrate_shape_function_gradient_in_domain = gaussIntegrateShapeFunctionGradientToReferenceCoordinateInParticleDomain(multi_corner_idx,initial_particle_domain);
        else
            approximate_integrate_shape_function_gradient_in_domain = gaussIntegrateShapeFunctionGradientToCurrentCoordinateInParticleDomain(multi_corner_idx,particle_domain,initial_particle_domain);
        for(InfluenceIterator iter(grid,particle_domain_vec[flat_corner_idx],weight_function); iter.valid(); ++node_num,++iter)
        {
            Vector<unsigned int,3> node_idx = iter.nodeIndex();
            unsigned int node_idx_1d = this->flatIndex(node_idx,grid.nodeNum());
            Vector<Scalar,3> corner_to_node = particle_domain_vec[flat_corner_idx] - grid.node(node_idx);
            for(unsigned int dim = 0; dim < 3; ++dim)
                corner_to_node[dim] /= grid_dx[dim];
            Scalar corner_weight = weight_function.weight(corner_to_node);
            //weight correspond to this node for domain corners
            corner_grid_weight[flat_corner_idx][node_num].node_idx_ = node_idx;
            corner_grid_weight[flat_corner_idx][node_num].weight_value_ = corner_weight;
            ++corner_grid_pair_num[flat_corner_idx];
            //weight and gradient correspond to this node for particles
            typename std::map<unsigned int,Scalar>::iterator weight_map_iter = idx_weight_map.find(node_idx_1d);
            if(weight_map_iter != idx_weight_map.end())
            {
                weight_map_iter->second += 1.0/domain_volume*approximate_integrate_shape_function_in_domain*corner_weight;
                idx_gradient_map[node_idx_1d] += 1.0/domain_volume*approximate_integrate_shape_function_gradient_in_domain*corner_weight;
            }
            else
            {
                idx_weight_map.insert(std::make_pair(node_idx_1d,1.0/domain_volume*approximate_integrate_shape_function_in_domain*corner_weight));
                idx_gradient_map.insert(std::make_pair(node_idx_1d,1.0/domain_volume*approximate_integrate_shape_function_gradient_in_domain*corner_weight));
            }
        }
    }
    //then store the data with respect to grid nodes
    particle_grid_pair_num = 0;
    for(typename std::map<unsigned int,Scalar>::iterator iter = idx_weight_map.begin(); iter != idx_weight_map.end(); ++iter)
    {
        if(iter->second > std::numeric_limits<Scalar>::epsilon()) //ignore nodes that have zero weight value, assume positive weight
        {
            particle_grid_weight_and_gradient[particle_grid_pair_num].node_idx_ = this->multiDimIndex(iter->first,grid.nodeNum());
            particle_grid_weight_and_gradient[particle_grid_pair_num].weight_value_ = iter->second;
            particle_grid_weight_and_gradient[particle_grid_pair_num].gradient_value_ = idx_gradient_map[iter->first];
            ++particle_grid_pair_num;
        }
    }
}

template <typename Scalar>
Scalar CPDI2UpdateMethod<Scalar,3>::gaussIntegrateShapeFunctionValueInParticleDomain(const Vector<unsigned int,3> &corner_idx, const ArrayND<Vector<Scalar,3>,3> &particle_domain)
{
    Scalar result = 0;
    // 2x2x2 gauss integration points
    Scalar one_over_sqrt_3 = 1.0/sqrt(3.0);
    Vector<Scalar,3> gauss_point;
    SquareMatrix<Scalar,3> jacobian;
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            for(unsigned int k = 0; k < 2; ++k)
            {
                gauss_point[0] = (2.0*i-1)*one_over_sqrt_3;
                gauss_point[1] = (2.0*j-1)*one_over_sqrt_3;
                gauss_point[2] = (2.0*k-1)*one_over_sqrt_3;
                jacobian = particleDomainJacobian(gauss_point,particle_domain);
                Scalar shape_function = 0.125*(1+(2.0*corner_idx[0]-1)*gauss_point[0])*(1+(2.0*corner_idx[1]-1)*gauss_point[1])*(1+(2.0*corner_idx[2]-1)*gauss_point[2]);
                result += shape_function*jacobian.determinant();
            }
    return result;
}

template <typename Scalar>
Vector<Scalar,3> CPDI2UpdateMethod<Scalar,3>::gaussIntegrateShapeFunctionGradientToCurrentCoordinateInParticleDomain(const Vector<unsigned int,3> &corner_idx, 
                                                                                                                     const ArrayND<Vector<Scalar,3>,3> &particle_domain,
                                                                                                                     const ArrayND<Vector<Scalar,3>,3> &initial_particle_domain)
{
    Vector<Scalar,3> result(0);
    //2x2x2 gauss integration points
    Scalar one_over_sqrt_3 = 1.0/sqrt(3.0);
    SquareMatrix<Scalar,3> jacobian, jacobian_inv_trans, ref_jacobian;
    Vector<Scalar,3> gauss_point, shape_function_derivative;
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            for(unsigned int k = 0; k < 2; ++k)
            {
                gauss_point[0] = (2.0*i-1)*one_over_sqrt_3;
                gauss_point[1] = (2.0*j-1)*one_over_sqrt_3;
                gauss_point[2] = (2.0*k-1)*one_over_sqrt_3;
                jacobian = particleDomainJacobian(gauss_point,particle_domain);
                jacobian_inv_trans = jacobian.inverse().transpose();
                ref_jacobian = particleDomainJacobian(gauss_point,initial_particle_domain);
                Scalar ref_jacobian_det = ref_jacobian.determinant();
                shape_function_derivative[0] = 0.125*(2.0*corner_idx[0]-1)*(1+(2.0*corner_idx[1]-1)*gauss_point[1])*(1+(2.0*corner_idx[2]-1)*gauss_point[2]);
                shape_function_derivative[1] = 0.125*(1+(2.0*corner_idx[0]-1)*gauss_point[0])*(2.0*corner_idx[1]-1)*(1+(2.0*corner_idx[2]-1)*gauss_point[2]);
                shape_function_derivative[2] = 0.125*(1+(2.0*corner_idx[0]-1)*gauss_point[0])*(1+(2.0*corner_idx[1]-1)*gauss_point[1])*(2.0*corner_idx[2]-1);
                result += jacobian_inv_trans*shape_function_derivative*ref_jacobian_det;
            }
    return result;
}

template <typename Scalar>
Vector<Scalar,3> CPDI2UpdateMethod<Scalar,3>::gaussIntegrateShapeFunctionGradientToReferenceCoordinateInParticleDomain(const Vector<unsigned int,3> &corner_idx,
                                                                                                                     const ArrayND<Vector<Scalar,3>,3> &initial_particle_domain)
{
    Vector<Scalar,3> result(0);
    //2x2x2 gauss integration points
    Scalar one_over_sqrt_3 = 1.0/sqrt(3.0);
    SquareMatrix<Scalar,3> ref_jacobian, ref_jacobian_inv_trans;
    Vector<Scalar,3> gauss_point, shape_function_derivative;
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            for(unsigned int k = 0; k < 2; ++k)
            {
                gauss_point[0] = (2.0*i-1)*one_over_sqrt_3;
                gauss_point[1] = (2.0*j-1)*one_over_sqrt_3;
                gauss_point[2] = (2.0*k-1)*one_over_sqrt_3;
                ref_jacobian = particleDomainJacobian(gauss_point,initial_particle_domain);
                ref_jacobian_inv_trans = ref_jacobian.inverse().transpose();
                Scalar ref_jacobian_det = ref_jacobian.determinant();
                shape_function_derivative[0] = 0.125*(2.0*corner_idx[0]-1)*(1+(2.0*corner_idx[1]-1)*gauss_point[1])*(1+(2.0*corner_idx[2]-1)*gauss_point[2]);
                shape_function_derivative[1] = 0.125*(1+(2.0*corner_idx[0]-1)*gauss_point[0])*(2.0*corner_idx[1]-1)*(1+(2.0*corner_idx[2]-1)*gauss_point[2]);
                shape_function_derivative[2] = 0.125*(1+(2.0*corner_idx[0]-1)*gauss_point[0])*(1+(2.0*corner_idx[1]-1)*gauss_point[1])*(2.0*corner_idx[2]-1);
                result += ref_jacobian_inv_trans*shape_function_derivative*ref_jacobian_det;
            }
    return result;
}

template <typename Scalar>
SquareMatrix<Scalar,3> CPDI2UpdateMethod<Scalar,3>::particleDomainJacobian(const Vector<Scalar,3> &eval_point, const ArrayND<Vector<Scalar,3>,3> &particle_domain)
{
    PHYSIKA_ASSERT(eval_point[0]>=-1&&eval_point[0]<=1);
    PHYSIKA_ASSERT(eval_point[1]>=-1&&eval_point[1]<=1);
    PHYSIKA_ASSERT(eval_point[2]>=-1&&eval_point[2]<=1);
    SquareMatrix<Scalar,3> jacobian(0);
    Vector<unsigned int,3> ele_idx;
    Vector<Scalar,3> domain_corner;
    for(typename ArrayND<Vector<Scalar,3>,3>::ConstIterator iter = particle_domain.begin(); iter != particle_domain.end(); ++iter)
    {
        ele_idx = iter.elementIndex();
        domain_corner = *iter;
        for(unsigned int row = 0; row < 3; ++row)
        {
            jacobian(row,0) += 0.125*(2.0*ele_idx[0]-1)*(1+(2.0*ele_idx[1]-1)*eval_point[1])*(1+(2.0*ele_idx[2]-1)*eval_point[2])*domain_corner[row];
            jacobian(row,1) += 0.125*(1+(2.0*ele_idx[0]-1)*eval_point[0])*(2.0*ele_idx[1]-1)*(1+(2.0*ele_idx[2]-1)*eval_point[2])*domain_corner[row];
            jacobian(row,2) += 0.125*(1+(2.0*ele_idx[0]-1)*eval_point[0])*(1+(2.0*ele_idx[1]-1)*eval_point[1])*(2.0*ele_idx[2]-1)*domain_corner[row];
        }
    }
    return jacobian;
}

template <typename Scalar>
Scalar CPDI2UpdateMethod<Scalar,3>::particleDomainVolume(const ArrayND<Vector<Scalar,3>,3> &particle_domain)
{
    Scalar volume = 0;
    //approximate volume of particle domain via integration of 1 inside domain
    //2x2x2 gauss integration points
    Scalar one_over_sqrt_3 = 1.0/sqrt(3.0);
    Vector<Scalar,3> gauss_point;
    SquareMatrix<Scalar,3> jacobian;
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            for(unsigned int k = 0; k < 2; ++k)
            {
                gauss_point[0] = (2.0*i-1)*one_over_sqrt_3;
                gauss_point[1] = (2.0*j-1)*one_over_sqrt_3;
                gauss_point[2] = (2.0*k-1)*one_over_sqrt_3;
                jacobian = particleDomainJacobian(gauss_point,particle_domain);
                volume += 1.0*jacobian.determinant();
            }
    return volume;
}

//explicit instantiations
template class CPDI2UpdateMethod<float,2>;
template class CPDI2UpdateMethod<double,2>;
template class CPDI2UpdateMethod<float,3>;
template class CPDI2UpdateMethod<double,3>;

}  //end of namespace Physika
