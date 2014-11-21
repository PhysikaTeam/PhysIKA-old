/*
 * @file mpm_solid_subgrid_friction_contact_method.cpp 
 * @Brief an algorithm that can resolve contact between mpm solids with subgrid resolution,
 *        the contact can be no-slip/free-slip with Coulomb friction model
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
#include <cmath>
#include <limits>
#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Geometry/Cartesian_Grids/grid.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/MPM/mpm_solid.h"
#include "Physika_Dynamics/MPM/MPM_Contact_Methods/mpm_solid_subgrid_friction_contact_method.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::MPMSolidSubgridFrictionContactMethod()
    :MPMSolidContactMethod<Scalar,Dim>(),friction_coefficient_(0.5),collide_threshold_(0.5)
{
}

template <typename Scalar, int Dim>
MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::MPMSolidSubgridFrictionContactMethod(const MPMSolidSubgridFrictionContactMethod<Scalar,Dim> &contact_method)
    :friction_coefficient_(contact_method.friction_coefficient_),
    collide_threshold_(contact_method.collide_threshold_)
{
    this->mpm_driver_ = contact_method.mpm_driver_;
}

template <typename Scalar, int Dim>
MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::~MPMSolidSubgridFrictionContactMethod()
{
}

template <typename Scalar, int Dim>
MPMSolidSubgridFrictionContactMethod<Scalar,Dim>& MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::operator=
(const MPMSolidSubgridFrictionContactMethod<Scalar,Dim> &contact_method)
{
    this->mpm_driver_ = contact_method.mpm_driver_;
    this->friction_coefficient_ = contact_method.friction_coefficient_;
    this->collide_threshold_ = contact_method.collide_threshold_;
    return *this;
}

template <typename Scalar, int Dim>
MPMSolidSubgridFrictionContactMethod<Scalar,Dim>* MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::clone() const
{
    return new MPMSolidSubgridFrictionContactMethod<Scalar,Dim>(*this);
}

template <typename Scalar, int Dim>
void MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::resolveContact(const std::vector<Vector<unsigned int,Dim> > &potential_collide_nodes,
                                                                      const std::vector<std::vector<unsigned int> > &objects_at_node,
                                                                      const std::vector<std::vector<Vector<Scalar,Dim> > > &normal_at_node,
                                                                      Scalar dt)
{
    MPMSolid<Scalar,Dim> *mpm_solid_driver = dynamic_cast<MPMSolid<Scalar,Dim>*>(this->mpm_driver_);
    if(mpm_solid_driver == NULL)
    {
        std::cerr<<"Error: mpm driver and contact method mismatch, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    //first init the particle buckets of the involved objects
    std::set<unsigned int> involved_objects;
    for(unsigned int i = 0; i < objects_at_node.size(); ++i)
        for(unsigned int j = 0; j < objects_at_node[i].size(); ++j)
            involved_objects.insert(objects_at_node[i][j]);
    ArrayND<std::map<unsigned int,std::vector<unsigned int> >,Dim> particle_bucket;
    initParticleBucket(involved_objects,particle_bucket);
    //resolve contact
    const Grid<Scalar,Dim> &grid = mpm_solid_driver->grid();
    for(unsigned int i = 0; i <potential_collide_nodes.size(); ++i)
    {
        Vector<unsigned int,Dim> node_idx = potential_collide_nodes[i];
        Vector<Scalar,Dim> node_pos = grid.node(node_idx);
        //first compute the center of mass velocity
        Vector<Scalar,Dim> vel_com(0);
        Scalar mass_com = 0;
        for(unsigned int j = 0; j < objects_at_node[i].size(); ++j)
        {
            unsigned int obj_idx = objects_at_node[i][j];
            Scalar mass = mpm_solid_driver->gridMass(obj_idx,node_idx);
            Vector<Scalar,Dim> velocity = mpm_solid_driver->gridVelocity(obj_idx,node_idx);
            vel_com += mass*velocity;
            mass_com += mass;
        }
        vel_com /= mass_com;
        //resolve contact for each object at the node
        for(unsigned int j = 0; j < objects_at_node[i].size(); ++j)
        {
            unsigned int obj_idx = objects_at_node[i][j];
            Vector<Scalar,Dim> trial_vel = mpm_solid_driver->gridVelocity(obj_idx,node_idx); //the velocity on grid that is solved independently
            Vector<Scalar,Dim> vel_delta =  trial_vel - vel_com;
            Scalar vel_delta_dot_norm = vel_delta.dot(normal_at_node[i][j]);
            if(vel_delta_dot_norm > 0)  //objects apporaching each other
            {
                //approximate the distance from the grid to the object surface with the minimum distance to the particles in adjacent cells
                std::vector<Vector<unsigned int,Dim> > adjacent_cells;
                adjacentCells(node_idx,adjacent_cells);
                Scalar min_dist = (std::numeric_limits<Scalar>::max)();
                for(unsigned int k = 0; k < adjacent_cells.size(); ++k)
                {
                    Vector<unsigned int,Dim> cell_idx = adjacent_cells[k];
                    std::vector<unsigned int> &particles_in_cell = particle_bucket(cell_idx)[obj_idx];
                    for(unsigned int l = 0; l < particles_in_cell.size(); ++l)
                    {
                        const SolidParticle<Scalar,Dim> &particle = mpm_solid_driver->particle(obj_idx,particles_in_cell[l]); 
                        Scalar dist = (node_pos - particle.position()).norm();
                        if(dist < min_dist)
                            min_dist = dist;
                    }
                }
                Scalar dist_threshold = collide_threshold_ * grid.minEdgeLength();
                if(min_dist < dist_threshold)
                {
                    //compute the tangential direction
                    Vector<Scalar,Dim> tangent_dir = tangentialDirection(normal_at_node[i][j],vel_delta);
                    //velocity difference in normal direction and tangential direction
                    Scalar vel_delta_dot_tan = vel_delta.dot(tangent_dir);
                    Vector<Scalar,Dim> vel_delta_norm = vel_delta_dot_norm * normal_at_node[i][j];
                    Vector<Scalar,Dim> vel_delta_tan = vel_delta_dot_tan * tangent_dir;
                    if(abs(vel_delta_dot_tan) > friction_coefficient_ * abs(vel_delta_dot_norm)) //slip with friction
                        vel_delta_tan = friction_coefficient_ * vel_delta_norm;
                    //apply a penalty function in the normal direction
                    Scalar penalty_factor = 1 - pow(min_dist/dist_threshold,penalty_power_);
                    vel_delta_norm *= penalty_factor;
                    //update the grid velocity
                    Vector<Scalar,Dim> new_vel = trial_vel - vel_delta_norm - vel_delta_tan;
                    mpm_solid_driver->setGridVelocity(obj_idx,node_idx,new_vel);
                }
            }
        }
    }
}

template <typename Scalar, int Dim>
void MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::setFrictionCoefficient(Scalar coefficient)
{
    if(coefficient < 0)
    {
        std::cout<<"Warning: invalid friction coefficient, 0.5 is used instead!\n";
        friction_coefficient_ = 0.5;
    }
    else
        friction_coefficient_ = coefficient;
}

template <typename Scalar, int Dim>
void MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::setCollideThreshold(Scalar threshold)
{
    if(threshold <= 0)
    {
        std::cout<<"Warning: invalid collide threshold, 0.5 of the grid cell edge length is used instead!\n";
        collide_threshold_ = 0.5;
    }
    else if(threshold > 1)
    {
        std::cout<<"Warning: collide threshold clamped to the cell size of grid!\n";
        collide_threshold_ = 1;
    }
    else
        collide_threshold_ = threshold;
}

template <typename Scalar, int Dim>
void MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::setPenaltyPower(Scalar penalty_power)
{
    if(penalty_power < 0)
    {
        std::cout<<"Warning: invalid penalty, 6 is used instead!\n";
        penalty_power_ = 6;
    }
    else
        penalty_power_ = penalty_power;
}
    
template <typename Scalar, int Dim>
Scalar MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::frictionCoefficient() const
{
    return friction_coefficient_;
}

template <typename Scalar, int Dim>
Scalar MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::collideThreshold() const
{
    return collide_threshold_;
}

template <typename Scalar, int Dim>
Scalar MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::penaltyPower() const
{
    return penalty_power_;
}
    
template <typename Scalar, int Dim>
void MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::initParticleBucket(const std::set<unsigned int> &objects, ArrayND<std::map<unsigned int, std::vector<unsigned int> >,Dim> &bucket) const
{
    MPMSolid<Scalar,Dim> *mpm_solid_driver = dynamic_cast<MPMSolid<Scalar,Dim>*>(this->mpm_driver_);
    PHYSIKA_ASSERT(mpm_solid_driver);
    const Grid<Scalar,Dim> &grid = mpm_solid_driver->grid();
    Vector<unsigned int,Dim> grid_cell_num = grid.cellNum();
    bucket.resize(grid_cell_num);
    for(std::set<unsigned int>::iterator iter = objects.begin(); iter != objects.end(); ++iter)
    {
        unsigned int obj_idx = *iter;
        for(unsigned int particle_idx = 0; particle_idx < mpm_solid_driver->particleNumOfObject(obj_idx); ++particle_idx)
        {
            const SolidParticle<Scalar,Dim> &particle = mpm_solid_driver->particle(obj_idx,particle_idx);
            Vector<Scalar,Dim> particle_pos = particle.position();
            Vector<unsigned int,Dim>  bucket_idx;
            Vector<Scalar,Dim> bias_in_cell;
            grid.cellIndexAndBiasInCell(particle_pos,bucket_idx,bias_in_cell);
            std::map<unsigned int,std::vector<unsigned int> >::iterator map_iter = bucket(bucket_idx).find(obj_idx);
            if(map_iter == bucket(bucket_idx).end())
            {
                std::vector<unsigned int> vec(1,particle_idx);
                bucket(bucket_idx).insert(std::make_pair(obj_idx,vec));
            }
            else
                bucket(bucket_idx)[obj_idx].push_back(particle_idx);
        }
    }
}

template <typename Scalar, int Dim>
Vector<Scalar,2> MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::tangentialDirection(const Vector<Scalar,2> &normal, const Vector<Scalar,2> &velocity_diff) const
{
//TO DO
    return Vector<Scalar,2>(0);
}

template <typename Scalar, int Dim>
Vector<Scalar,3> MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::tangentialDirection(const Vector<Scalar,3> &normal, const Vector<Scalar,3> &velocity_diff) const
{
    Vector<Scalar,3> tangent_dir = velocity_diff.cross(normal);
    tangent_dir.normalize();
    tangent_dir = normal.cross(tangent_dir);
    return tangent_dir;
}

template <typename Scalar, int Dim>
void MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::adjacentCells(const Vector<unsigned int,Dim> &node_idx, std::vector<Vector<unsigned int,Dim> > &cells) const
{
    cells.clear();
    Vector<unsigned int,Dim> cell_idx = node_idx;
    switch(Dim)
    {
    case 2:
    {
        for(int offset_x = -1; offset_x <= 0; ++offset_x)
            for(int offset_y = -1; offset_y <= 0; ++offset_y)
            {
                cell_idx[0] += offset_x;
                cell_idx[1] += offset_y;
                if(cell_idx[0] < 0 || cell_idx[1] < 0)
                    continue;
                cells.push_back(cell_idx);
            }
        break;
    }
    case 3:
    {
        for(int offset_x = -1; offset_x <= 0; ++offset_x)
            for(int offset_y = -1; offset_y <= 0; ++offset_y)
                for(int offset_z = -1; offset_z <= 0; ++offset_z)
            {
                cell_idx[0] += offset_x;
                cell_idx[1] += offset_y;
                cell_idx[2] += offset_z;
                if(cell_idx[0] < 0 || cell_idx[1] < 0 || cell_idx[2] < 0)
                    continue;
                cells.push_back(cell_idx);
            }
        break;
    }
    default:
        PHYSIKA_ERROR("Wrong dimension specified!");
    }
}

//explicit instantiations
template class MPMSolidSubgridFrictionContactMethod<float,2>;
template class MPMSolidSubgridFrictionContactMethod<float,3>;
template class MPMSolidSubgridFrictionContactMethod<double,2>;
template class MPMSolidSubgridFrictionContactMethod<double,3>;

}  //end of namespace Physika
