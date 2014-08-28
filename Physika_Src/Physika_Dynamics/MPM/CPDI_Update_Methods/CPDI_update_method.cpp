/*
 * @file CPDI_update_method.cpp 
 * @Brief the particle domain update procedure introduced in paper:
 *        "A convected particle domain interpolation technique to extend applicability of
 *         the material point method for problems involving massive deformations"
 *        It's the base class of all update methods derived from CPDI
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

#include <cstdlib>
#include <iostream>
#include <map>
#include "Physika_Core/Arrays/array_Nd.h"
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Grid_Weight_Functions/grid_weight_function.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Geometry/Cartesian_Grids/grid.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/MPM/Weight_Function_Influence_Iterators/uniform_grid_weight_function_influence_iterator.h"
#include "Physika_Dynamics/MPM/CPDI_mpm_solid.h"
#include "Physika_Dynamics/MPM/CPDI_Update_Methods/CPDI_update_method.h"

namespace Physika{

template <typename Scalar>
CPDIUpdateMethod<Scalar,2>::CPDIUpdateMethod()
    :cpdi_driver_(NULL)
{
}

template <typename Scalar>
CPDIUpdateMethod<Scalar,2>::~CPDIUpdateMethod()
{
}

template <typename Scalar>
void CPDIUpdateMethod<Scalar,2>::updateParticleInterpolationWeight(const GridWeightFunction<Scalar,2> &weight_function,
                                                                   std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > &particle_grid_weight_and_gradient,
                                                                   std::vector<unsigned int> &particle_grid_pair_num)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    for(unsigned int i = 0; i < this->cpdi_driver_->particleNum(); ++i)
        updateParticleInterpolationWeight(i,weight_function,particle_grid_weight_and_gradient[i],particle_grid_pair_num[i]);
}

template <typename Scalar>
void CPDIUpdateMethod<Scalar,2>::updateParticleDomain()
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    ArrayND<Vector<Scalar,2>,2> particle_domain;
    for(unsigned int i = 0; i < cpdi_driver_->particleNum(); ++i)
    {
        SquareMatrix<Scalar,2> deform_grad = (cpdi_driver_->particle(i)).deformationGradient();
        cpdi_driver_->initialParticleDomain(i,particle_domain);
        Vector<Scalar,2> particle_pos = (cpdi_driver_->particle(i)).position();
        Vector<unsigned int,2> corner_idx(0);
        Vector<Scalar,2> min_corner = particle_domain(corner_idx);
        corner_idx[0] = 1;
        Vector<Scalar,2> x_corner = particle_domain(corner_idx);
        corner_idx[0] = 0; corner_idx[1] = 1;
        Vector<Scalar,2> y_corner = particle_domain(corner_idx);
        Vector<Scalar,2> r_x = x_corner - min_corner;
        Vector<Scalar,2> r_y = y_corner - min_corner;
        //update parallelogram
        r_x = deform_grad * r_x;
        r_y = deform_grad * r_y;
        //update 4 corners
        min_corner = particle_pos - 0.5*r_x - 0.5*r_y;
        for(unsigned int idx_x = 0; idx_x < 2; ++idx_x)
            for(unsigned int idx_y = 0; idx_y < 2; ++idx_y)
            {
                corner_idx[0] = idx_x;
                corner_idx[1] = idx_y;
                particle_domain(corner_idx) = min_corner + idx_x*r_x + idx_y*r_y;
            }
        cpdi_driver_->setCurrentParticleDomain(i,particle_domain);
    }
}

template <typename Scalar>
void CPDIUpdateMethod<Scalar,2>::setCPDIDriver(CPDIMPMSolid<Scalar,2> *cpdi_driver)
{
    if(cpdi_driver==NULL)
    {
        std::cerr<<"Error: Cannot set NULL CPDI driver to CPDIUpdateMethod, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    this->cpdi_driver_ = cpdi_driver;
}

template <typename Scalar>
void CPDIUpdateMethod<Scalar,2>::updateParticleInterpolationWeight(unsigned int particle_idx, const GridWeightFunction<Scalar,2> &weight_function,
                                                                   std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > &particle_grid_weight_and_gradient,
                                                                   unsigned int &particle_grid_pair_num)
{
    ArrayND<Vector<Scalar,2>,2> particle_domain;
    this->cpdi_driver_->currentParticleDomain(particle_idx,particle_domain);
    const SolidParticle<Scalar,2> &particle = this->cpdi_driver_->particle(particle_idx);
    std::map<unsigned int,Scalar> idx_weight_map;
    std::map<unsigned int,Vector<Scalar,2> > idx_gradient_map;
    const Grid<Scalar,2> &grid = this->cpdi_driver_->grid();
    Vector<Scalar,2> r_x = particle_domain(Vector<unsigned int,2>(1,0)) - particle_domain(Vector<unsigned int,2>(0));
    Vector<Scalar,2> r_y = particle_domain(Vector<unsigned int,2>(0,1)) - particle_domain(Vector<unsigned int,2>(0));
    Vector<Scalar,2> grid_dx = grid.dX();
    typedef UniformGridWeightFunctionInfluenceIterator<Scalar,2> InfluenceIterator;
    //first compute the weight and gradient with respect to each grid node in the influence range
    for(typename ArrayND<Vector<Scalar,2>,2>::Iterator corner_iter = particle_domain.begin(); corner_iter != particle_domain.end(); ++corner_iter)
    {
        Vector<unsigned int,2> corner_idx = corner_iter.elementIndex();
        unsigned int flat_corner_idx = flatIndex(corner_idx,Vector<unsigned int,2>(2));
        for(InfluenceIterator iter(grid,particle_domain(corner_idx),weight_function); iter.valid(); ++iter)
        {
            Vector<unsigned int,2> node_idx = iter.nodeIndex();
            unsigned int node_idx_1d = flatIndex(node_idx,grid.nodeNum());
            Vector<Scalar,2> corner_to_node = particle_domain(corner_idx) - grid.node(node_idx);
            for(unsigned int dim = 0; dim < 2; ++dim)
                corner_to_node[dim] /= grid_dx[dim];
            Scalar corner_weight = weight_function.weight(corner_to_node);
            //weight corresond to this node
            typename std::map<unsigned int,Scalar>::iterator weight_map_iter = idx_weight_map.find(node_idx_1d);
            if(weight_map_iter != idx_weight_map.end())
                weight_map_iter->second += 0.25*corner_weight;
            else
				idx_weight_map.insert(std::make_pair(node_idx_1d,0.25*corner_weight));
            //gradient correspond to this node
            typename std::map<unsigned int,Vector<Scalar,2> >::iterator gradient_map_iter = idx_gradient_map.find(node_idx_1d);
            switch(flat_corner_idx)
            {
            case 0:
            {
                if(gradient_map_iter != idx_gradient_map.end())
                {
                    gradient_map_iter->second[0] += 1.0/(2.0*particle.volume())*corner_weight*(r_x[1]-r_y[1]);
                    gradient_map_iter->second[1] += 1.0/(2.0*particle.volume())*corner_weight*(-r_x[0]+r_y[0]);
                }
                else
                {
					Vector<Scalar,2> gradient;
                    gradient[0] = 1.0/(2.0*particle.volume())*corner_weight*(r_x[1]-r_y[1]);
                    gradient[1] = 1.0/(2.0*particle.volume())*corner_weight*(-r_x[0]+r_y[0]);
					idx_gradient_map.insert(std::make_pair(node_idx_1d,gradient));
                }
                break;
            }
            case 1:
            {
                if(gradient_map_iter != idx_gradient_map.end())
                {
                    gradient_map_iter->second[0] += -1.0/(2.0*particle.volume())*corner_weight*(r_x[1]+r_y[1]);
                    gradient_map_iter->second[1] += -1.0/(2.0*particle.volume())*corner_weight*(-r_x[0]-r_y[0]);
                }
                else
                {
					Vector<Scalar,2> gradient;
                    gradient[0] = -1.0/(2.0*particle.volume())*corner_weight*(r_x[1]+r_y[1]);
                    gradient[1] = -1.0/(2.0*particle.volume())*corner_weight*(-r_x[0]-r_y[0]);
					idx_gradient_map.insert(std::make_pair(node_idx_1d,gradient));
                }
                break;
            }
            case 2:
            {
                if(gradient_map_iter != idx_gradient_map.end())
                {
                    gradient_map_iter->second[0] += 1.0/(2.0*particle.volume())*corner_weight*(r_x[1]+r_y[1]);
                    gradient_map_iter->second[1] += 1.0/(2.0*particle.volume())*corner_weight*(-r_x[0]-r_y[0]);
                }
                else
                {
					Vector<Scalar,2> gradient;
                    gradient[0] = 1.0/(2.0*particle.volume())*corner_weight*(r_x[1]+r_y[1]);
                    gradient[1] = 1.0/(2.0*particle.volume())*corner_weight*(-r_x[0]-r_y[0]);
					idx_gradient_map.insert(std::make_pair(node_idx_1d,gradient));
                }
                break;
            }
            case 3:
            {
                if(gradient_map_iter != idx_gradient_map.end())
                {
                    gradient_map_iter->second[0] += -1.0/(2.0*particle.volume())*corner_weight*(r_x[1]-r_y[1]);
                    gradient_map_iter->second[1] += -1.0/(2.0*particle.volume())*corner_weight*(-r_x[0]+r_y[0]);
                }
                else
                {
					Vector<Scalar,2> gradient;
                    gradient[0] = -1.0/(2.0*particle.volume())*corner_weight*(r_x[1]-r_y[1]);
                    gradient[1] = -1.0/(2.0*particle.volume())*corner_weight*(-r_x[0]+r_y[0]);
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
        particle_grid_weight_and_gradient[particle_grid_pair_num].node_idx_ = multiDimIndex(iter->first,grid.nodeNum());
        particle_grid_weight_and_gradient[particle_grid_pair_num].weight_value_ = iter->second;
        particle_grid_weight_and_gradient[particle_grid_pair_num].gradient_value_ = idx_gradient_map[iter->first];
        ++particle_grid_pair_num;
    }
}

template <typename Scalar>
unsigned int CPDIUpdateMethod<Scalar,2>::flatIndex(const Vector<unsigned int,2> &index, const Vector<unsigned int,2> &dimension) const
{
    unsigned int flat_index = 0;
    Vector<unsigned int,2> vec = index;
    for(unsigned int i = 0; i < 2; ++i)
    {
        for(unsigned int j = i+1; j < 2; ++j)
            vec[i] *= dimension[j];
        flat_index += vec[i];
    }
    return flat_index;
}

template <typename Scalar>
Vector<unsigned int,2> CPDIUpdateMethod<Scalar,2>::multiDimIndex(unsigned int flat_index, const Vector<unsigned int,2> &dimension) const
{
    Vector<unsigned int,2> index(1);
    for(unsigned int i = 0; i < 2; ++i)
    {
        for(unsigned int j = i+1; j < 2; ++j)
            index[i] *= dimension[j];
        unsigned int temp = flat_index / index[i];
        flat_index = flat_index % index[i];
        index[i] = temp;
    }
    return index;
}

///////////////////////////////////////////////////// 3D ///////////////////////////////////////////////////

template <typename Scalar>
CPDIUpdateMethod<Scalar,3>::CPDIUpdateMethod()
    :cpdi_driver_(NULL)
{
}

template <typename Scalar>
CPDIUpdateMethod<Scalar,3>::~CPDIUpdateMethod()
{
}

template <typename Scalar>
void CPDIUpdateMethod<Scalar,3>::updateParticleInterpolationWeight(const GridWeightFunction<Scalar,3> &weight_function,
                                                                   std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > &particle_grid_weight_and_gradient,
                                                                   std::vector<unsigned int> &particle_grid_pair_num)
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    for(unsigned int i = 0; i < this->cpdi_driver_->particleNum(); ++i)
        updateParticleInterpolationWeight(i,weight_function,particle_grid_weight_and_gradient[i],particle_grid_pair_num[i]);
}

template <typename Scalar>
void CPDIUpdateMethod<Scalar,3>::updateParticleDomain()
{
    PHYSIKA_ASSERT(this->cpdi_driver_);
    //TO DO
}

template <typename Scalar>
void CPDIUpdateMethod<Scalar,3>::setCPDIDriver(CPDIMPMSolid<Scalar,3> *cpdi_driver)
{
    if(cpdi_driver==NULL)
    {
        std::cerr<<"Error: Cannot set NULL CPDI driver to CPDIUpdateMethod, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    this->cpdi_driver_ = cpdi_driver;
}

template <typename Scalar>
void CPDIUpdateMethod<Scalar,3>::updateParticleInterpolationWeight(unsigned int particle_idx, const GridWeightFunction<Scalar,3> &weight_function,
                                                                   std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > &particle_grid_weight_and_gradient,
                                                                   unsigned int &particle_grid_pair_num)
{
//TO DO
}

template <typename Scalar>
unsigned int CPDIUpdateMethod<Scalar,3>::flatIndex(const Vector<unsigned int,3> &index, const Vector<unsigned int,3> &dimension) const
{
    unsigned int flat_index = 0;
    Vector<unsigned int,3> vec = index;
    for(unsigned int i = 0; i < 3; ++i)
    {
        for(unsigned int j = i+1; j < 3; ++j)
            vec[i] *= dimension[j];
        flat_index += vec[i];
    }
    return flat_index;
}

template <typename Scalar>
Vector<unsigned int,3> CPDIUpdateMethod<Scalar,3>::multiDimIndex(unsigned int flat_index, const Vector<unsigned int,3> &dimension) const
{
    Vector<unsigned int,3> index(1);
    for(unsigned int i = 0; i < 3; ++i)
    {
        for(unsigned int j = i+1; j < 3; ++j)
            index[i] *= dimension[j];
        unsigned int temp = flat_index / index[i];
        flat_index = flat_index % index[i];
        index[i] = temp;
    }
    return index;
}

//explicit instantiations
template class CPDIUpdateMethod<float,2>;
template class CPDIUpdateMethod<double,2>;
template class CPDIUpdateMethod<float,3>;
template class CPDIUpdateMethod<double,3>;

}  //end of namespace Physika
