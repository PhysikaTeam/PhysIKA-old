/*
 * @file robust_CPDI2_update_method.cpp 
 * @brief enhanced version of the particle domain update procedure introduced in paper:
 *        "Second-order convected particle domain interpolation with enrichment for weak
 *         discontinuities at material interfaces"
 *    We made some key modifications(enhancements) to the conventional CPDI2 to improve
 *    its robustness with degenerated particle domain during simulation
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <limits>
#include <map>
#include <algorithm>
#include "Physika_Core/Arrays/array_Nd.h"
#include "Physika_Core/Grid_Weight_Functions/grid_weight_function.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Geometry/Cartesian_Grids/grid.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/Utilities/Grid_Weight_Function_Influence_Iterators/uniform_grid_weight_function_influence_iterator.h"
#include "Physika_Dynamics/MPM/CPDI_mpm_solid.h"
#include "Physika_Dynamics/MPM/CPDI_Update_Methods/robust_CPDI2_update_method.h"

namespace Physika{

template <typename Scalar>
RobustCPDI2UpdateMethod<Scalar,2>::RobustCPDI2UpdateMethod()
    :CPDI2UpdateMethod<Scalar,2>()
{
}

template <typename Scalar>
RobustCPDI2UpdateMethod<Scalar,2>::~RobustCPDI2UpdateMethod()
{
}

template <typename Scalar>
void RobustCPDI2UpdateMethod<Scalar,2>::updateParticleInterpolationWeight(const GridWeightFunction<Scalar,2> &weight_function,
                                  std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > > &particle_grid_weight_and_gradient,
                                  std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
                                  std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,2> > > > > &corner_grid_weight,
                                  std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    for(unsigned int i = 0; i < this->cpdi_driver_->objectNum(); ++i)
        for(unsigned int j = 0; j < this->cpdi_driver_->particleNumOfObject(i); ++j)
            updateParticleInterpolationWeight(i,j,weight_function,particle_grid_weight_and_gradient[i][j],particle_grid_pair_num[i][j],
                                              corner_grid_weight[i][j],corner_grid_pair_num[i][j]);
}

template <typename Scalar>
void RobustCPDI2UpdateMethod<Scalar,2>::updateParticleInterpolationWeight(const GridWeightFunction<Scalar,2> &weight_function,
                                  const std::vector<VolumetricMesh<Scalar,2>*> &particle_domain_mesh,
                                  std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > > &particle_grid_weight_and_gradient,
                                  std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
                                  std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,2> > > > > &corner_grid_weight,
                                  std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num)
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
        //avoid redundant computation of corner grid weight
        unsigned int global_corner_num = particle_domain_mesh[object_idx]->vertNum();
        global_corner_grid_pair_num.resize(global_corner_num);
        std::fill(global_corner_grid_pair_num.begin(), global_corner_grid_pair_num.end(), 0);//use 0 to indicate uninitialized 
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
                gradient_integral = gaussIntegrateShapeFunctionGradientInParticleDomain(multi_corner_idx,initial_particle_domain);
                Vector<Scalar,2> particle_corner_gradient = 1.0/domain_volume*gradient_integral;
                //now compute weight/gradient between particles and grid nodes, corners and grid nodes
                unsigned int global_corner_idx = particle_domain_mesh[object_idx]->eleVertIndex(particle_idx,flat_corner_idx);
                if(global_corner_grid_pair_num[global_corner_idx] > 0) //weight between this corner and grid nodes has been computed before
                {
                    corner_grid_pair_num[object_idx][particle_idx][flat_corner_idx] = global_corner_grid_pair_num[global_corner_idx];
                    for(unsigned int corner_grid_pair_idx = 0; corner_grid_pair_idx < global_corner_grid_pair_num[global_corner_idx]; ++corner_grid_pair_idx)
                    {
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][corner_grid_pair_idx] = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx];
                        Scalar corner_weight = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx].weight_value;
                        Vector<unsigned int,2> node_idx = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx].node_idx;
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
                    global_corner_grid_weight[global_corner_idx].clear();
                    for(InfluenceIterator iter(grid,particle_domain_vec[flat_corner_idx],weight_function); iter.valid(); ++node_num,++iter)
                    {
                        Vector<unsigned int,2> node_idx = iter.nodeIndex();
                        unsigned int node_idx_1d = this->flatIndex(node_idx,grid_node_num);
                        Vector<Scalar,2> corner_to_node = particle_domain_vec[flat_corner_idx] - grid.node(node_idx);
                        for(unsigned int dim = 0; dim < 2; ++dim)
                            corner_to_node[dim] /= grid_dx[dim];
                        Scalar corner_weight = weight_function.weight(corner_to_node);
                        //weight and gradient correspond to this node for domain corners
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][node_num].node_idx = node_idx;
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][node_num].weight_value = corner_weight;
                        ++corner_grid_pair_num[object_idx][particle_idx][flat_corner_idx];
                        //store the computed weight between domain corners and grid nodes to avoid redundant computation for other particles
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
                    unsigned int this_particle_grid_pair_num = particle_grid_pair_num[object_idx][particle_idx];
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].node_idx = this->multiDimIndex(iter->first,grid_node_num);
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].weight_value = iter->second;
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].gradient_value = idx_gradient_map[iter->first];
                    ++particle_grid_pair_num[object_idx][particle_idx];
                }
            }
        }
    }
}

template <typename Scalar>
void RobustCPDI2UpdateMethod<Scalar,2>::updateParticleInterpolationWeightWithEnrichment(const GridWeightFunction<Scalar,2> &weight_function,
                                  const std::vector<VolumetricMesh<Scalar,2>*> &particle_domain_mesh,
                                  const std::vector<std::vector<unsigned char> > &is_enriched_domain_corner,
                                  std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > > &particle_grid_weight_and_gradient,
                                  std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
                                  std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,2> > > > > &corner_grid_weight,
                                  std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num)
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
        //avoid redundant computation of corner grid weight
        unsigned int global_corner_num = particle_domain_mesh[object_idx]->vertNum();
        global_corner_grid_pair_num.resize(global_corner_num);
        std::fill(global_corner_grid_pair_num.begin(), global_corner_grid_pair_num.end(), 0);//use 0 to indicate uninitialized 
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
                gradient_integral = gaussIntegrateShapeFunctionGradientInParticleDomain(multi_corner_idx,initial_particle_domain);
                Vector<Scalar,2> particle_corner_gradient = 1.0/domain_volume*gradient_integral;
                //now compute weight/gradient between particles and grid nodes, corners and grid nodes
                unsigned int global_corner_idx = particle_domain_mesh[object_idx]->eleVertIndex(particle_idx,flat_corner_idx);
                if(global_corner_grid_pair_num[global_corner_idx] > 0) //weight between this corner and grid nodes has been computed before
                {
                    corner_grid_pair_num[object_idx][particle_idx][flat_corner_idx] = global_corner_grid_pair_num[global_corner_idx];
                    for(unsigned int corner_grid_pair_idx = 0; corner_grid_pair_idx < global_corner_grid_pair_num[global_corner_idx]; ++corner_grid_pair_idx)
                    {
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][corner_grid_pair_idx] = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx];
                        Scalar corner_weight = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx].weight_value;
                        Vector<unsigned int,2> node_idx = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx].node_idx;
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
                    global_corner_grid_weight[global_corner_idx].clear();
                    for(InfluenceIterator iter(grid,particle_domain_vec[flat_corner_idx],weight_function); iter.valid(); ++node_num,++iter)
                    {
                        Vector<unsigned int,2> node_idx = iter.nodeIndex();
                        unsigned int node_idx_1d = this->flatIndex(node_idx,grid_node_num);
                        Vector<Scalar,2> corner_to_node = particle_domain_vec[flat_corner_idx] - grid.node(node_idx);
                        for(unsigned int dim = 0; dim < 2; ++dim)
                            corner_to_node[dim] /= grid_dx[dim];
                        Scalar corner_weight = weight_function.weight(corner_to_node);
                        //weight and gradient correspond to this node for domain corners
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][node_num].node_idx = node_idx;
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][node_num].weight_value = corner_weight;
                        ++corner_grid_pair_num[object_idx][particle_idx][flat_corner_idx];
                        //store the computed weight between domain corners and grid nodes to avoid redundant computation for other particles
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
                    unsigned int this_particle_grid_pair_num = particle_grid_pair_num[object_idx][particle_idx];
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].node_idx = this->multiDimIndex(iter->first,grid_node_num);
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].weight_value = iter->second;
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].gradient_value = idx_gradient_map[iter->first];
                    ++particle_grid_pair_num[object_idx][particle_idx];
                }
            }
        }
    }
}

template <typename Scalar>
SquareMatrix<Scalar,2> RobustCPDI2UpdateMethod<Scalar,2>::computeParticleDeformationGradientFromDomainShape(unsigned int obj_idx, unsigned int particle_idx)
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
            jacobian = this->particleDomainJacobian(gauss_point,initial_particle_domain);
            particle_deform_grad += this->computeDeformationGradientAtPointInParticleDomain(obj_idx,particle_idx,gauss_point)*jacobian.determinant();
        }
    //average
    particle_deform_grad /= domain_volume;
    return particle_deform_grad;
}

template <typename Scalar>
void RobustCPDI2UpdateMethod<Scalar,2>::computeParticleInterpolationWeightInParticleDomain(unsigned int obj_idx, unsigned int particle_idx, std::vector<Scalar> &particle_corner_weight)
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
void RobustCPDI2UpdateMethod<Scalar,2>::computeParticleInterpolationGradientInParticleDomain(unsigned int obj_idx, unsigned int particle_idx,
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
            particle_corner_gradient[flat_corner_idx] = 1.0/domain_volume*gaussIntegrateShapeFunctionGradientInParticleDomain(corner_idx,initial_particle_domain);
        }
}

template <typename Scalar>
void RobustCPDI2UpdateMethod<Scalar,2>::updateParticleInterpolationWeight(unsigned int object_idx, unsigned int particle_idx, const GridWeightFunction<Scalar,2> &weight_function,
                                  std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > &particle_grid_weight_and_gradient,
                                  unsigned int &particle_grid_pair_num,
                                  std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,2>  > > &corner_grid_weight,
                                  std::vector<unsigned int> &corner_grid_pair_num)
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
        gradient_integral = gaussIntegrateShapeFunctionGradientInParticleDomain(corner_idx, initial_particle_domain);
        for(InfluenceIterator iter(grid,particle_domain_vec[flat_corner_idx],weight_function); iter.valid(); ++node_num,++iter)
        {
            Vector<unsigned int,2> node_idx = iter.nodeIndex();
            unsigned int node_idx_1d = this->flatIndex(node_idx,grid.nodeNum());
            Vector<Scalar,2> corner_to_node = particle_domain_vec[flat_corner_idx] - grid.node(node_idx);
            for(unsigned int dim = 0; dim < 2; ++dim)
                corner_to_node[dim] /= grid_dx[dim];
            Scalar corner_weight = weight_function.weight(corner_to_node);
            //weight correspond to this node for domain corners
            corner_grid_weight[flat_corner_idx][node_num].node_idx = node_idx;
            corner_grid_weight[flat_corner_idx][node_num].weight_value = corner_weight;
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
            particle_grid_weight_and_gradient[particle_grid_pair_num].node_idx = this->multiDimIndex(iter->first,grid.nodeNum());
            particle_grid_weight_and_gradient[particle_grid_pair_num].weight_value = iter->second;
            particle_grid_weight_and_gradient[particle_grid_pair_num].gradient_value = idx_gradient_map[iter->first];
            ++particle_grid_pair_num;
        }
    }
}

template <typename Scalar>
Vector<Scalar,2> RobustCPDI2UpdateMethod<Scalar,2>::gaussIntegrateShapeFunctionGradientInParticleDomain(
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
                ref_jacobian = this->particleDomainJacobian(gauss_point,initial_particle_domain);
                ref_jacobian_inv_trans = ref_jacobian.inverse().transpose();
                Scalar ref_jacobian_det = ref_jacobian.determinant();                
                shape_function_derivative[0] = 0.25*(2.0*corner_idx[0]-1)*(1+(2.0*corner_idx[1]-1)*gauss_point[1]);
                shape_function_derivative[1] = 0.25*(1+(2.0*corner_idx[0]-1)*gauss_point[0])*(2.0*corner_idx[1]-1);
                result += ref_jacobian_inv_trans*shape_function_derivative*ref_jacobian_det;
            }
    return result;
}

///////////////////////////////////////////////////// 3D ///////////////////////////////////////////////////

template <typename Scalar>
RobustCPDI2UpdateMethod<Scalar,3>::RobustCPDI2UpdateMethod()
    :CPDI2UpdateMethod<Scalar,3>()
{
}

template <typename Scalar>
RobustCPDI2UpdateMethod<Scalar,3>::~RobustCPDI2UpdateMethod()
{
}

template <typename Scalar>
void RobustCPDI2UpdateMethod<Scalar,3>::updateParticleInterpolationWeight(const GridWeightFunction<Scalar,3> &weight_function,
                                 std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > > &particle_grid_weight_and_gradient,
                                 std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
                                 std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,3> > > > > &corner_grid_weight,
                                 std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    for(unsigned int i = 0; i < this->cpdi_driver_->objectNum(); ++i)
        for(unsigned int j = 0; j < this->cpdi_driver_->particleNumOfObject(i); ++j)
            updateParticleInterpolationWeight(i,j,weight_function,particle_grid_weight_and_gradient[i][j],particle_grid_pair_num[i][j],
                                              corner_grid_weight[i][j],corner_grid_pair_num[i][j]);
}

template <typename Scalar>
void RobustCPDI2UpdateMethod<Scalar,3>::updateParticleInterpolationWeight(const GridWeightFunction<Scalar,3> &weight_function,
                                 const std::vector<VolumetricMesh<Scalar,3>*> &particle_domain_mesh,
                                 std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > > &particle_grid_weight_and_gradient,
                                 std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
                                 std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,3> > > > > &corner_grid_weight,
                                 std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num)
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
        //avoid redundant computation of corner grid weight
        unsigned int global_corner_num = particle_domain_mesh[object_idx]->vertNum();
        global_corner_grid_pair_num.resize(global_corner_num);
        std::fill(global_corner_grid_pair_num.begin(), global_corner_grid_pair_num.end(), 0);//use 0 to indicate uninitialized 
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
            Scalar domain_volume = this->particleDomainVolume(initial_particle_domain);

            //first compute the weight and gradient with respect to each grid node in the influence range of the particle
            //node weight and gradient with respect to domain corners are stored as well
            for(flat_corner_idx = 0; flat_corner_idx < particle_domain_vec.size(); ++flat_corner_idx)
            {
                Vector<unsigned int,3> multi_corner_idx = this->multiDimIndex(flat_corner_idx,Vector<unsigned int,3>(2));
                Scalar approximate_integrate_shape_function_in_domain = gaussIntegrateShapeFunctionValueInParticleDomain(multi_corner_idx,initial_particle_domain);
                Vector<Scalar,3> approximate_integrate_shape_function_gradient_in_domain(0);
                approximate_integrate_shape_function_gradient_in_domain = gaussIntegrateShapeFunctionGradientInParticleDomain(multi_corner_idx,initial_particle_domain);
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
                        Scalar corner_weight = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx].weight_value;
                        Vector<unsigned int,3> node_idx = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx].node_idx;
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
                    global_corner_grid_weight[global_corner_idx].clear();
                    for(InfluenceIterator iter(grid,particle_domain_vec[flat_corner_idx],weight_function); iter.valid(); ++node_num,++iter)
                    {
                        Vector<unsigned int,3> node_idx = iter.nodeIndex();
                        unsigned int node_idx_1d = this->flatIndex(node_idx,grid_node_num);
                        Vector<Scalar,3> corner_to_node = particle_domain_vec[flat_corner_idx] - grid.node(node_idx);
                        for(unsigned int dim = 0; dim < 3; ++dim)
                            corner_to_node[dim] /= grid_dx[dim];
                        Scalar corner_weight = weight_function.weight(corner_to_node);
                        //weight correspond to this node for domain corners
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][node_num].node_idx = node_idx;
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][node_num].weight_value = corner_weight;
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
                    unsigned int this_particle_grid_pair_num = particle_grid_pair_num[object_idx][particle_idx];
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].node_idx = this->multiDimIndex(iter->first,grid_node_num);
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].weight_value = iter->second;
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].gradient_value = idx_gradient_map[iter->first];
                    ++particle_grid_pair_num[object_idx][particle_idx];
                }
            }
        }
    }
}

template <typename Scalar>
void RobustCPDI2UpdateMethod<Scalar,3>::updateParticleInterpolationWeightWithEnrichment(const GridWeightFunction<Scalar,3> &weight_function,
                                  const std::vector<VolumetricMesh<Scalar,3>*> &particle_domain_mesh,
                                  const std::vector<std::vector<unsigned char> > &is_enriched_domain_corner,
                                  std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > > &particle_grid_weight_and_gradient,
                                  std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
                                  std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,3>  > > > > &corner_grid_weight,
                                  std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num)
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
        global_corner_grid_pair_num.resize(global_corner_num);
        std::fill(global_corner_grid_pair_num.begin(), global_corner_grid_pair_num.end(), 0);//use 0 to indicate uninitialized 
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
            Scalar domain_volume = this->particleDomainVolume(initial_particle_domain);
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
                approximate_integrate_shape_function_gradient_in_domain = gaussIntegrateShapeFunctionGradientInParticleDomain(multi_corner_idx,initial_particle_domain);
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
                        Scalar corner_weight = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx].weight_value;
                        Vector<unsigned int,3> node_idx = global_corner_grid_weight[global_corner_idx][corner_grid_pair_idx].node_idx;
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
                    global_corner_grid_weight[global_corner_idx].clear();
                    for(InfluenceIterator iter(grid,particle_domain_vec[flat_corner_idx],weight_function); iter.valid(); ++node_num,++iter)
                    {
                        Vector<unsigned int,3> node_idx = iter.nodeIndex();
                        unsigned int node_idx_1d = this->flatIndex(node_idx,grid_node_num);
                        Vector<Scalar,3> corner_to_node = particle_domain_vec[flat_corner_idx] - grid.node(node_idx);
                        for(unsigned int dim = 0; dim < 3; ++dim)
                            corner_to_node[dim] /= grid_dx[dim];
                        Scalar corner_weight = weight_function.weight(corner_to_node);
                        //weight correspond to this node for domain corners
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][node_num].node_idx = node_idx;
                        corner_grid_weight[object_idx][particle_idx][flat_corner_idx][node_num].weight_value = corner_weight;
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
                    unsigned int this_particle_grid_pair_num = particle_grid_pair_num[object_idx][particle_idx];
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].node_idx = this->multiDimIndex(iter->first,grid_node_num);
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].weight_value = iter->second;
                    particle_grid_weight_and_gradient[object_idx][particle_idx][this_particle_grid_pair_num].gradient_value = idx_gradient_map[iter->first];
                    ++particle_grid_pair_num[object_idx][particle_idx];
                }
            }
        }
    }
}

template <typename Scalar>
SquareMatrix<Scalar,3> RobustCPDI2UpdateMethod<Scalar,3>::computeParticleDeformationGradientFromDomainShape(unsigned int obj_idx, unsigned int particle_idx)
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
                jacobian = this->particleDomainJacobian(gauss_point,initial_particle_domain);
                particle_deform_grad += this->computeDeformationGradientAtPointInParticleDomain(obj_idx,particle_idx,gauss_point)*jacobian.determinant();
            }
    Scalar particle_domain_volume = this->particleDomainVolume(initial_particle_domain);
    PHYSIKA_ASSERT(particle_domain_volume > 0);
    return particle_deform_grad/particle_domain_volume;
}

template <typename Scalar>
void RobustCPDI2UpdateMethod<Scalar,3>::computeParticleInterpolationWeightInParticleDomain(unsigned int obj_idx, unsigned int particle_idx, std::vector<Scalar> &particle_corner_weight)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    PHYSIKA_ASSERT(obj_idx < this->cpdi_driver_->objectNum());
    PHYSIKA_ASSERT(particle_idx < this->cpdi_driver_->particleNumOfObject(obj_idx));
    PHYSIKA_ASSERT(particle_corner_weight.size() >= 8);
    ArrayND<Vector<Scalar,3>,3> initial_particle_domain;
    this->cpdi_driver_->initialParticleDomain(obj_idx,particle_idx,initial_particle_domain);
    Scalar particle_domain_volume = this->particleDomainVolume(initial_particle_domain);
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
void RobustCPDI2UpdateMethod<Scalar,3>::computeParticleInterpolationGradientInParticleDomain(unsigned int obj_idx, unsigned int particle_idx,
                                                                                                                                              std::vector<Vector<Scalar,3> > &particle_corner_gradient)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    PHYSIKA_ASSERT(obj_idx < this->cpdi_driver_->objectNum());
    PHYSIKA_ASSERT(particle_idx < this->cpdi_driver_->particleNumOfObject(obj_idx));
    PHYSIKA_ASSERT(particle_corner_gradient.size() >= 8);
    ArrayND<Vector<Scalar,3>,3> initial_particle_domain;
    this->cpdi_driver_->initialParticleDomain(obj_idx,particle_idx,initial_particle_domain);
    Scalar particle_domain_volume = this->particleDomainVolume(initial_particle_domain);
    Vector<unsigned int,3> corner_idx(0);
    //eight corners
    unsigned int corner_idx_1d = 0;
    for(corner_idx[0] = 0; corner_idx[0] < 2; ++corner_idx[0])
        for(corner_idx[1] = 0; corner_idx[1] < 2; ++corner_idx[1])
            for(corner_idx[2] = 0; corner_idx[2] < 2; ++corner_idx[2])
            {
                //integrate
                particle_corner_gradient[corner_idx_1d] = this->gaussIntegrateShapeFunctionGradientInParticleDomain(corner_idx,initial_particle_domain);
                //average
                particle_corner_gradient[corner_idx_1d] /= particle_domain_volume;
                ++corner_idx_1d;
            }
}

template <typename Scalar>
void RobustCPDI2UpdateMethod<Scalar,3>::updateParticleInterpolationWeight(unsigned int object_idx, unsigned int particle_idx, const GridWeightFunction<Scalar,3> &weight_function,
                                                                    std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > &particle_grid_weight_and_gradient,
                                                                    unsigned int &particle_grid_pair_num,
                                                                    std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,3> > > &corner_grid_weight,
                                                                    std::vector<unsigned int> &corner_grid_pair_num)
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
    Scalar domain_volume = this->particleDomainVolume(initial_particle_domain);

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
        approximate_integrate_shape_function_gradient_in_domain = gaussIntegrateShapeFunctionGradientInParticleDomain(multi_corner_idx,initial_particle_domain);
        for(InfluenceIterator iter(grid,particle_domain_vec[flat_corner_idx],weight_function); iter.valid(); ++node_num,++iter)
        {
            Vector<unsigned int,3> node_idx = iter.nodeIndex();
            unsigned int node_idx_1d = this->flatIndex(node_idx,grid.nodeNum());
            Vector<Scalar,3> corner_to_node = particle_domain_vec[flat_corner_idx] - grid.node(node_idx);
            for(unsigned int dim = 0; dim < 3; ++dim)
                corner_to_node[dim] /= grid_dx[dim];
            Scalar corner_weight = weight_function.weight(corner_to_node);
            //weight correspond to this node for domain corners
            corner_grid_weight[flat_corner_idx][node_num].node_idx = node_idx;
            corner_grid_weight[flat_corner_idx][node_num].weight_value = corner_weight;
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
            particle_grid_weight_and_gradient[particle_grid_pair_num].node_idx = this->multiDimIndex(iter->first,grid.nodeNum());
            particle_grid_weight_and_gradient[particle_grid_pair_num].weight_value = iter->second;
            particle_grid_weight_and_gradient[particle_grid_pair_num].gradient_value = idx_gradient_map[iter->first];
            ++particle_grid_pair_num;
        }
    }
}

template <typename Scalar>
Scalar RobustCPDI2UpdateMethod<Scalar,3>::gaussIntegrateShapeFunctionValueInParticleDomain(const Vector<unsigned int,3> &corner_idx, const ArrayND<Vector<Scalar,3>,3> &particle_domain)
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
                jacobian = this->particleDomainJacobian(gauss_point,particle_domain);
                Scalar shape_function = 0.125*(1+(2.0*corner_idx[0]-1)*gauss_point[0])*(1+(2.0*corner_idx[1]-1)*gauss_point[1])*(1+(2.0*corner_idx[2]-1)*gauss_point[2]);
                result += shape_function*jacobian.determinant();
            }
    return result;
}

template <typename Scalar>
Vector<Scalar,3> RobustCPDI2UpdateMethod<Scalar,3>::gaussIntegrateShapeFunctionGradientInParticleDomain(const Vector<unsigned int,3> &corner_idx,
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
                ref_jacobian = this->particleDomainJacobian(gauss_point,initial_particle_domain);
                ref_jacobian_inv_trans = ref_jacobian.inverse().transpose();
                Scalar ref_jacobian_det = ref_jacobian.determinant();
                shape_function_derivative[0] = 0.125*(2.0*corner_idx[0]-1)*(1+(2.0*corner_idx[1]-1)*gauss_point[1])*(1+(2.0*corner_idx[2]-1)*gauss_point[2]);
                shape_function_derivative[1] = 0.125*(1+(2.0*corner_idx[0]-1)*gauss_point[0])*(2.0*corner_idx[1]-1)*(1+(2.0*corner_idx[2]-1)*gauss_point[2]);
                shape_function_derivative[2] = 0.125*(1+(2.0*corner_idx[0]-1)*gauss_point[0])*(1+(2.0*corner_idx[1]-1)*gauss_point[1])*(2.0*corner_idx[2]-1);
                result += ref_jacobian_inv_trans*shape_function_derivative*ref_jacobian_det;
            }
    return result;
}

//explicit instantiations
template class RobustCPDI2UpdateMethod<float,2>;
template class RobustCPDI2UpdateMethod<double,2>;
template class RobustCPDI2UpdateMethod<float,3>;
template class RobustCPDI2UpdateMethod<double,3>;

}  //end of namespace Physika
