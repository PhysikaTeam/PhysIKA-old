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
#include "Physika_Core/Utilities/math_utilities.h"
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
                                  std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > > &particle_grid_weight_and_gradient,
                                  std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
                                  std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > > > &corner_grid_weight_and_gradient,
                                  std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    for(unsigned int i = 0; i < this->cpdi_driver_->objectNum(); ++i)
        for(unsigned int j = 0; j < this->cpdi_driver_->particleNumOfObject(i); ++j)
            updateParticleInterpolationWeight(i,j,weight_function,particle_grid_weight_and_gradient[i][j],particle_grid_pair_num[i][j],
                                              corner_grid_weight_and_gradient[i][j],corner_grid_pair_num[i][j]);
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,2>::updateParticleInterpolationWeightInDomain(std::vector<std::vector<std::vector<Scalar> > > &particle_corner_weight,
                                                                            std::vector<std::vector<std::vector<Vector<Scalar,2> > > > &particle_corner_gradient)
{
//TO DO
    PHYSIKA_ASSERT(this->cpdi_driver_);
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,2>::updateParticleDomain(
    const std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > > > &corner_grid_weight_and_gradient,
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
                    Scalar weight = corner_grid_weight_and_gradient[obj_idx][particle_idx][corner_idx][j].weight_value_;
                    Vector<Scalar,2> node_vel = this->cpdi_driver_->gridVelocity(obj_idx,corner_grid_weight_and_gradient[obj_idx][particle_idx][corner_idx][j].node_idx_);
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
            this->cpdi_driver_->currentParticleDomain(obj_idx,particle_idx,particle_domain);
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
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,2>::updateParticleDeformationGradient()
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    ArrayND<Vector<Scalar,2>,2> particle_domain, initial_particle_domain;
    std::vector<Vector<Scalar,2> > particle_domain_vec(4), particle_domain_displacement(4);
    SquareMatrix<Scalar,2> identity = SquareMatrix<Scalar,2>::identityMatrix();
    for(unsigned int obj_idx = 0; obj_idx < this->cpdi_driver_->objectNum(); ++obj_idx)
    {
        for(unsigned int particle_idx = 0; particle_idx < this->cpdi_driver_->particleNumOfObject(obj_idx); ++particle_idx)
        {
            SolidParticle<Scalar,2> &particle = this->cpdi_driver_->particle(obj_idx,particle_idx);
            this->cpdi_driver_->currentParticleDomain(obj_idx,particle_idx,particle_domain);
            this->cpdi_driver_->initialParticleDomain(obj_idx,particle_idx,initial_particle_domain);
            unsigned int i = 0;
            for(typename ArrayND<Vector<Scalar,2>,2>::Iterator corner_iter = particle_domain.begin(); corner_iter != particle_domain.end(); ++i,++corner_iter)
                particle_domain_vec[i] = *corner_iter;
            i = 0;
            for(typename ArrayND<Vector<Scalar,2>,2>::Iterator corner_iter = initial_particle_domain.begin(); corner_iter != initial_particle_domain.end(); ++i,++corner_iter)
                particle_domain_displacement[i] = particle_domain_vec[i] - (*corner_iter);
            //coefficients
            Scalar a = (particle_domain_vec[2]-particle_domain_vec[0]).cross(particle_domain_vec[1]-particle_domain_vec[0]);
            Scalar b = (particle_domain_vec[2]-particle_domain_vec[0]).cross(particle_domain_vec[3]-particle_domain_vec[1]);
            Scalar c = (particle_domain_vec[3]-particle_domain_vec[2]).cross(particle_domain_vec[1]-particle_domain_vec[0]);
            Scalar domain_volume = a + 0.5*(b+c);
            SquareMatrix<Scalar,2> particle_deform_grad = identity;
            Vector<Scalar,2> gradient_integral;
            i = 0;
            for(typename ArrayND<Vector<Scalar,2>,2>::Iterator corner_iter = particle_domain.begin(); corner_iter != particle_domain.end(); ++i,++corner_iter)
            {
                Vector<unsigned int,2> corner_idx = corner_iter.elementIndex();
                gradient_integral = gaussIntegrateShapeFunctionGradientToReferenceCoordinateInParticleDomain(corner_idx,particle_domain,initial_particle_domain);
                particle_deform_grad += 1.0/domain_volume * particle_domain_displacement[i].outerProduct(gradient_integral);
            }
            //update particle deformation gradient
            particle.setDeformationGradient(particle_deform_grad);
        }
    }    
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,2>::updateParticleInterpolationWeight(unsigned int object_idx, unsigned int particle_idx, const GridWeightFunction<Scalar,2> &weight_function,
                                                                    std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > &particle_grid_weight_and_gradient,
                                                                    unsigned int &particle_grid_pair_num,
                                                                    std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > &corner_grid_weight_and_gradient,
                                                                    std::vector<unsigned int> &corner_grid_pair_num)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    ArrayND<Vector<Scalar,2>,2> particle_domain;
    this->cpdi_driver_->currentParticleDomain(object_idx,particle_idx,particle_domain);
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
Vector<Scalar,2> CPDI2UpdateMethod<Scalar,2>::gaussIntegrateShapeFunctionGradientToReferenceCoordinateInParticleDomain(const Vector<unsigned int,2> &corner_idx, 
                                                                                                        const ArrayND<Vector<Scalar,2>,2> &particle_domain,
                                                                                                        const ArrayND<Vector<Scalar,2>,2> &initial_particle_domain)
{
    Vector<Scalar,2> result(0);
    //2x2 gauss integration points
    Scalar one_over_sqrt_3 = 1.0/sqrt(3.0);
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
        {
            Vector<Scalar,2> gauss_point((2.0*i-1)*one_over_sqrt_3,(2.0*j-1)*one_over_sqrt_3);
            SquareMatrix<Scalar,2> jacobian = particleDomainJacobian(gauss_point,particle_domain);
            Scalar jacobian_det = jacobian.determinant();
            SquareMatrix<Scalar,2> ref_jacobian = particleDomainJacobian(gauss_point,initial_particle_domain);
            SquareMatrix<Scalar,2> ref_jacobian_inv_trans = ref_jacobian.inverse().transpose();
            Vector<Scalar,2> shape_function_derivative;
            shape_function_derivative[0] = 0.25*(2.0*corner_idx[0]-1)*(1+(2.0*corner_idx[1]-1)*gauss_point[1]);
            shape_function_derivative[1] = 0.25*(1+(2.0*corner_idx[0]-1)*gauss_point[0])*(2.0*corner_idx[1]-1);
            result += ref_jacobian_inv_trans*shape_function_derivative*jacobian_det;
        }
    return result;
}

template <typename Scalar>
SquareMatrix<Scalar,2> CPDI2UpdateMethod<Scalar,2>::particleDomainJacobian(const Vector<Scalar,2> &eval_point, const ArrayND<Vector<Scalar,2>,2> &particle_domain)
{
    PHYSIKA_ASSERT(eval_point[0]>=-1&&eval_point[0]<=1);
    PHYSIKA_ASSERT(eval_point[1]>=-1&&eval_point[1]<=1);
    SquareMatrix<Scalar,2> jacobian(0);
    for(typename ArrayND<Vector<Scalar,2>,2>::ConstIterator iter = particle_domain.begin(); iter != particle_domain.end(); ++iter)
    {
        Vector<unsigned int,2> ele_idx = iter.elementIndex();
        Vector<Scalar,2> domain_corner = *iter;
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
                                 std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > >  > > &corner_grid_weight_and_gradient,
                                 std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    for(unsigned int i = 0; i < this->cpdi_driver_->objectNum(); ++i)
        for(unsigned int j = 0; j < this->cpdi_driver_->particleNumOfObject(i); ++j)
            updateParticleInterpolationWeight(i,j,weight_function,particle_grid_weight_and_gradient[i][j],particle_grid_pair_num[i][j],
                                              corner_grid_weight_and_gradient[i][j],corner_grid_pair_num[i][j]);
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,3>::updateParticleInterpolationWeightInDomain(std::vector<std::vector<std::vector<Scalar> > > &particle_corner_weight,
                                                                            std::vector<std::vector<std::vector<Vector<Scalar,3> > > > &particle_corner_gradient)
{
//TO DO
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,3>::updateParticleDomain(
    const std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > > > &corner_grid_weight_and_gradient,
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
                    Scalar weight = corner_grid_weight_and_gradient[obj_idx][particle_idx][corner_idx][j].weight_value_;
                    Vector<Scalar,3> node_vel = this->cpdi_driver_->gridVelocity(obj_idx,corner_grid_weight_and_gradient[obj_idx][particle_idx][corner_idx][j].node_idx_);
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
            //TO DO: compute domain volume instead of using particle volume
            Scalar domain_volume = particle.volume();
            Vector<Scalar,3> new_pos(0);
            for(unsigned int flat_corner_idx = 0; flat_corner_idx < 8; ++flat_corner_idx)
            {
                Vector<unsigned int,3> multi_corner_idx = this->multiDimIndex(flat_corner_idx,Vector<unsigned int,3>(2));
                Scalar approximate_integrate_shape_function_in_domain = gaussIntegrateShapeFunctionValueInParticleDomain(multi_corner_idx,particle_domain);
                new_pos += 1.0/domain_volume*approximate_integrate_shape_function_in_domain*particle_domain_vec[flat_corner_idx];
            }
            particle.setPosition(new_pos);
        }
    }
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,3>::updateParticleDeformationGradient()
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    ArrayND<Vector<Scalar,3>,3> particle_domain, initial_particle_domain;
    std::vector<Vector<Scalar,3> > particle_domain_vec(8), particle_domain_displacement(8);
    SquareMatrix<Scalar,3> identity = SquareMatrix<Scalar,3>::identityMatrix();
    for(unsigned int obj_idx = 0; obj_idx < this->cpdi_driver_->objectNum(); ++obj_idx)
    {
        for(unsigned int particle_idx = 0; particle_idx < this->cpdi_driver_->particleNumOfObject(obj_idx); ++particle_idx)
        {
            SolidParticle<Scalar,3> &particle = this->cpdi_driver_->particle(obj_idx,particle_idx);
            this->cpdi_driver_->currentParticleDomain(obj_idx,particle_idx,particle_domain);
            this->cpdi_driver_->initialParticleDomain(obj_idx,particle_idx,initial_particle_domain);
            unsigned int i = 0;
            for(typename ArrayND<Vector<Scalar,3>,3>::Iterator corner_iter = particle_domain.begin(); corner_iter != particle_domain.end(); ++i,++corner_iter)
                particle_domain_vec[i] = *corner_iter;
            i = 0;
            for(typename ArrayND<Vector<Scalar,3>,3>::Iterator corner_iter = initial_particle_domain.begin(); corner_iter != initial_particle_domain.end(); ++i,++corner_iter)
                particle_domain_displacement[i] = particle_domain_vec[i] - (*corner_iter);
            //TO DO: compute domain volume instead of using particle volume
            Scalar domain_volume = particle.volume();
            SquareMatrix<Scalar,3> particle_deform_grad = identity;
            Vector<Scalar,3> gradient_integral;
            i = 0;
            for(typename ArrayND<Vector<Scalar,3>,3>::Iterator corner_iter = particle_domain.begin(); corner_iter != particle_domain.end(); ++i,++corner_iter)
            {
                Vector<unsigned int,3> corner_idx = corner_iter.elementIndex();
                gradient_integral = gaussIntegrateShapeFunctionGradientToReferenceCoordinateInParticleDomain(corner_idx,particle_domain,initial_particle_domain);
                particle_deform_grad += 1.0/domain_volume * particle_domain_displacement[i].outerProduct(gradient_integral);
            }
            //update particle deformation gradient
            particle.setDeformationGradient(particle_deform_grad);
        }
    }    
}

template <typename Scalar>
void CPDI2UpdateMethod<Scalar,3>::updateParticleInterpolationWeight(unsigned int object_idx, unsigned int particle_idx, const GridWeightFunction<Scalar,3> &weight_function,
                                                                    std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > &particle_grid_weight_and_gradient,
                                                                    unsigned int &particle_grid_pair_num,
                                                                    std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > &corner_grid_weight_and_gradient,
                                                                    std::vector<unsigned int> &corner_grid_pair_num)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    ArrayND<Vector<Scalar,3>,3> particle_domain;
    this->cpdi_driver_->currentParticleDomain(object_idx,particle_idx,particle_domain);
    std::vector<Vector<Scalar,3> > particle_domain_vec;
    for(typename ArrayND<Vector<Scalar,3>,3>::Iterator corner_iter = particle_domain.begin(); corner_iter != particle_domain.end(); ++corner_iter)
        particle_domain_vec.push_back(*corner_iter);
    std::map<unsigned int,Scalar> idx_weight_map;
    std::map<unsigned int,Vector<Scalar,3> > idx_gradient_map;
    const Grid<Scalar,3> &grid = this->cpdi_driver_->grid();
    Vector<Scalar,3> grid_dx = grid.dX();
    const SolidParticle<Scalar,3> &particle = this->cpdi_driver_->particle(object_idx,particle_idx);
    //TO DO: compute domain volume instead of using particle volume
    Scalar domain_volume = particle.volume();

    typedef UniformGridWeightFunctionInfluenceIterator<Scalar,3> InfluenceIterator;
    //first compute the weight and gradient with respect to each grid node in the influence range of the particle
    //node weight and gradient with respect to domain corners are stored as well
    for(unsigned int flat_corner_idx = 0; flat_corner_idx < particle_domain_vec.size(); ++flat_corner_idx)
    {
        corner_grid_pair_num[flat_corner_idx] = 0;
        unsigned int node_num = 0;
        Vector<unsigned int,3> multi_corner_idx = this->multiDimIndex(flat_corner_idx,Vector<unsigned int,3>(2));
        Scalar approximate_integrate_shape_function_in_domain = gaussIntegrateShapeFunctionValueInParticleDomain(multi_corner_idx,particle_domain);
        Vector<Scalar,3> approximate_integrate_shape_function_gradient_in_domain = gaussIntegrateShapeFunctionGradientToCurrentCoordinateInParticleDomain(multi_corner_idx,particle_domain);
        for(InfluenceIterator iter(grid,particle_domain_vec[flat_corner_idx],weight_function); iter.valid(); ++node_num,++iter)
        {
            Vector<unsigned int,3> node_idx = iter.nodeIndex();
            unsigned int node_idx_1d = this->flatIndex(node_idx,grid.nodeNum());
            Vector<Scalar,3> corner_to_node = particle_domain_vec[flat_corner_idx] - grid.node(node_idx);
            for(unsigned int dim = 0; dim < 3; ++dim)
                corner_to_node[dim] /= grid_dx[dim];
            Scalar corner_weight = weight_function.weight(corner_to_node);
            //weight and gradient correspond to this node for domain corners
            corner_grid_weight_and_gradient[flat_corner_idx][node_num].node_idx_ = node_idx;
            corner_grid_weight_and_gradient[flat_corner_idx][node_num].weight_value_ = corner_weight;
            corner_grid_weight_and_gradient[flat_corner_idx][node_num].gradient_value_ = weight_function.gradient(corner_to_node);
            ++corner_grid_pair_num[flat_corner_idx];
            //weight and gradient correspond to this node for particles
            typename std::map<unsigned int,Scalar>::iterator weight_map_iter = idx_weight_map.find(node_idx_1d);
            typename std::map<unsigned int,Vector<Scalar,3> >::iterator gradient_map_iter = idx_gradient_map.find(node_idx_1d);
            if(weight_map_iter != idx_weight_map.end())
                weight_map_iter->second += 1.0/domain_volume*approximate_integrate_shape_function_in_domain*corner_weight;
            else
                idx_weight_map.insert(std::make_pair(node_idx_1d,1.0/domain_volume*approximate_integrate_shape_function_in_domain*corner_weight));
            if(gradient_map_iter != idx_gradient_map.end())
                gradient_map_iter->second += 1.0/domain_volume*approximate_integrate_shape_function_gradient_in_domain*corner_weight;
            else
                idx_gradient_map.insert(std::make_pair(node_idx_1d,1.0/domain_volume*approximate_integrate_shape_function_gradient_in_domain*corner_weight));
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
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            for(unsigned int k = 0; k < 2; ++k)
            {
                Vector<Scalar,3> gauss_point((2.0*i-1)*one_over_sqrt_3,(2.0*j-1)*one_over_sqrt_3,(2.0*k-1)*one_over_sqrt_3);
                SquareMatrix<Scalar,3> jacobian = particleDomainJacobian(gauss_point,particle_domain);
                Scalar shape_function = 0.125*(1+(2.0*corner_idx[0]-1)*gauss_point[0])*(1+(2.0*corner_idx[1]-1)*gauss_point[1])*(1+(2.0*corner_idx[2]-1)*gauss_point[2]);
                result += shape_function*jacobian.determinant();
            }
    return result;
}

template <typename Scalar>
Vector<Scalar,3> CPDI2UpdateMethod<Scalar,3>::gaussIntegrateShapeFunctionGradientToCurrentCoordinateInParticleDomain(const Vector<unsigned int,3> &corner_idx, const ArrayND<Vector<Scalar,3>,3> &particle_domain)
{
    Vector<Scalar,3> result(0);
    //2x2x2 gauss integration points
    Scalar one_over_sqrt_3 = 1.0/sqrt(3.0);
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            for(unsigned int k = 0; k < 2; ++k)
            {
                Vector<Scalar,3> gauss_point((2.0*i-1)*one_over_sqrt_3,(2.0*j-1)*one_over_sqrt_3,(2.0*k-1)*one_over_sqrt_3);
                SquareMatrix<Scalar,3> jacobian = particleDomainJacobian(gauss_point,particle_domain);
                SquareMatrix<Scalar,3> jacobian_inv_trans = jacobian.inverse().transpose();
                Scalar jacobian_det = jacobian.determinant();
                Vector<Scalar,3> shape_function_derivative;
                shape_function_derivative[0] = 0.125*(2.0*corner_idx[0]-1)*(1+(2.0*corner_idx[1]-1)*gauss_point[1])*(1+(2.0*corner_idx[2]-1)*gauss_point[2]);
                shape_function_derivative[1] = 0.125*(1+(2.0*corner_idx[0]-1)*gauss_point[0])*(2.0*corner_idx[1]-1)*(1+(2.0*corner_idx[2]-1)*gauss_point[2]);
                shape_function_derivative[2] = 0.125*(1+(2.0*corner_idx[0]-1)*gauss_point[0])*(1+(2.0*corner_idx[1]-1)*gauss_point[1])*(2.0*corner_idx[2]-1);
                result += jacobian_inv_trans*shape_function_derivative*jacobian_det;
            }
    return result;
}

template <typename Scalar>
Vector<Scalar,3> CPDI2UpdateMethod<Scalar,3>::gaussIntegrateShapeFunctionGradientToReferenceCoordinateInParticleDomain(const Vector<unsigned int,3> &corner_idx,
                                                                                                                       const ArrayND<Vector<Scalar,3>,3> &particle_domain,
                                                                                                                       const ArrayND<Vector<Scalar,3>,3> &initial_particle_domain)
{
    Vector<Scalar,3> result(0);
    //2x2x2 gauss integration points
    Scalar one_over_sqrt_3 = 1.0/sqrt(3.0);
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            for(unsigned int k = 0; k < 2; ++k)
            {
                Vector<Scalar,3> gauss_point((2.0*i-1)*one_over_sqrt_3,(2.0*j-1)*one_over_sqrt_3,(2.0*k-1)*one_over_sqrt_3);
                SquareMatrix<Scalar,3> jacobian = particleDomainJacobian(gauss_point,particle_domain);
                Scalar jacobian_det = jacobian.determinant();
                SquareMatrix<Scalar,3> ref_jacobian = particleDomainJacobian(gauss_point,initial_particle_domain);
                SquareMatrix<Scalar,3> ref_jacobian_inv_trans = ref_jacobian.inverse().transpose();
                Vector<Scalar,3> shape_function_derivative;
                shape_function_derivative[0] = 0.125*(2.0*corner_idx[0]-1)*(1+(2.0*corner_idx[1]-1)*gauss_point[1])*(1+(2.0*corner_idx[2]-1)*gauss_point[2]);
                shape_function_derivative[1] = 0.125*(1+(2.0*corner_idx[0]-1)*gauss_point[0])*(2.0*corner_idx[1]-1)*(1+(2.0*corner_idx[2]-1)*gauss_point[2]);
                shape_function_derivative[2] = 0.125*(1+(2.0*corner_idx[0]-1)*gauss_point[0])*(1+(2.0*corner_idx[1]-1)*gauss_point[1])*(2.0*corner_idx[2]-1);
                result += ref_jacobian_inv_trans*shape_function_derivative*jacobian_det;
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
    for(typename ArrayND<Vector<Scalar,3>,3>::ConstIterator iter = particle_domain.begin(); iter != particle_domain.end(); ++iter)
    {
        Vector<unsigned int,3> ele_idx = iter.elementIndex();
        Vector<Scalar,3> domain_corner = *iter;
        for(unsigned int row = 0; row < 3; ++row)
        {
            jacobian(row,0) += 0.125*(2.0*ele_idx[0]-1)*(1+(2.0*ele_idx[1]-1)*eval_point[1])*(1+(2.0*ele_idx[2]-1)*eval_point[2])*domain_corner[row];
            jacobian(row,1) += 0.125*(1+(2.0*ele_idx[0]-1)*eval_point[0])*(2.0*ele_idx[1]-1)*(1+(2.0*ele_idx[2]-1)*eval_point[2])*domain_corner[row];
            jacobian(row,2) += 0.125*(1+(2.0*ele_idx[0]-1)*eval_point[0])*(1+(2.0*ele_idx[1]-1)*eval_point[1])*(2.0*ele_idx[2]-1)*domain_corner[row];
        }
    }
    return jacobian;
}

//explicit instantiations
template class CPDI2UpdateMethod<float,2>;
template class CPDI2UpdateMethod<double,2>;
template class CPDI2UpdateMethod<float,3>;
template class CPDI2UpdateMethod<double,3>;

}  //end of namespace Physika
