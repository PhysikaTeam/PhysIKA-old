/*
 * @file mpm_solid_subgrid_friction_contact_method.cpp 
 * @Brief an algorithm that can resolve contact between multiple mpm solids with subgrid resolution,
 *        the contact can be no-slip/free-slip with Coulomb friction model
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

#include <cstdlib>
#include <cmath>
#include <limits>
#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Geometry/Cartesian_Grids/grid.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/MPM/mpm_solid.h"
#include "Physika_Dynamics/MPM/MPM_Contact_Methods/mpm_solid_subgrid_friction_contact_method.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::MPMSolidSubgridFrictionContactMethod()
    :MPMSolidContactMethod<Scalar,Dim>(),
    friction_coefficient_(0.5),
    collide_threshold_(0.5),
    penalty_power_(6),
    restitution_coefficient_(0)
{
}

template <typename Scalar, int Dim>
MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::MPMSolidSubgridFrictionContactMethod(const MPMSolidSubgridFrictionContactMethod<Scalar,Dim> &contact_method)
    :friction_coefficient_(contact_method.friction_coefficient_),
     collide_threshold_(contact_method.collide_threshold_),
     penalty_power_(contact_method.penalty_power_),
     restitution_coefficient_(contact_method.restitution_coefficient_)
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
    this->penalty_power_ = contact_method.penalty_power_;
    this->restitution_coefficient_ = contact_method.restitution_coefficient_;
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
                                                                      const std::vector<std::vector<unsigned char> > &is_dirichlet_at_node,
                                                                      Scalar dt)
{
    MPMSolid<Scalar,Dim> *mpm_solid_driver = dynamic_cast<MPMSolid<Scalar,Dim>*>(this->mpm_driver_);
    if(mpm_solid_driver == NULL)
        throw PhysikaException("mpm driver and contact method mismatch!");
    if(potential_collide_nodes.empty()) //no collision
        return;
    //init particle bucket for all involved objects
    std::set<unsigned int> involved_objects;
    for(unsigned int i = 0; i < potential_collide_nodes.size(); ++i)
        for(unsigned int j = 0; j < objects_at_node[i].size(); ++j)
            involved_objects.insert(objects_at_node[i][j]);
    initParticleBucket(involved_objects,particle_bucket_);
    //resolve contact at each node
    for(unsigned int i = 0; i < potential_collide_nodes.size(); ++i)
    {
        Vector<unsigned int,Dim> node_idx = potential_collide_nodes[i];
        std::vector<Vector<Scalar,Dim> > velocity_delta(objects_at_node[i].size(),Vector<Scalar,Dim>(0));  //velocity impulse for each object at the node
        for(unsigned int j = 0; j < objects_at_node[i].size(); ++j)
            for(unsigned int k = j + 1; k < objects_at_node[i].size(); ++k)
            {
                Vector<Scalar,Dim> obj1_vel_delta(0), obj2_vel_delta(0);
                resolveContactBetweenTwoObjects(node_idx,objects_at_node[i][j],objects_at_node[i][k],
                                                normal_at_node[i][j],normal_at_node[i][k],
                                                is_dirichlet_at_node[i][j],is_dirichlet_at_node[i][k],dt,
                                                obj1_vel_delta,obj2_vel_delta);
                velocity_delta[j] += obj1_vel_delta;
                velocity_delta[k] += obj2_vel_delta;
            }
        for(unsigned int j = 0; j < objects_at_node[i].size(); ++j)
        {
            Vector<Scalar,Dim> new_vel = mpm_solid_driver->gridVelocity(objects_at_node[i][j],node_idx) + velocity_delta[j];
            mpm_solid_driver->setGridVelocity(objects_at_node[i][j],node_idx,new_vel);
        }
    }
}

template <typename Scalar, int Dim>
void MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::setFrictionCoefficient(Scalar coefficient)
{
    if(coefficient < 0)
    {
        std::cerr<<"Warning: invalid friction coefficient, 0.5 is used instead!\n";
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
        std::cerr<<"Warning: invalid collide threshold, 0.5 of the grid cell edge length is used instead!\n";
        collide_threshold_ = 0.5;
    }
    else if(threshold > 1)
    {
        std::cerr<<"Warning: collide threshold clamped to the cell size of grid!\n";
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
        std::cerr<<"Warning: invalid penalty, 6 is used instead!\n";
        penalty_power_ = 6;
    }
    else
        penalty_power_ = penalty_power;
}
    
template <typename Scalar, int Dim>
void MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::setRestitutionCoefficient(Scalar restitution_coefficient)
{
    if(restitution_coefficient < 0)
    {
        std::cerr<<"Warning: invalid restitution, 0 (inelastic contact) is used instead!\n";
        restitution_coefficient_ = 0;
    }
    else if(restitution_coefficient > 1)
    {
        std::cerr<<"Warning: invalid restitution, 1 (elastic contact) is used instead!\n";
        restitution_coefficient_ = 1;
    }
    else
        restitution_coefficient_ = restitution_coefficient;
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
Scalar MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::restitutionCoefficient() const
{
    return restitution_coefficient_;
}
    
template <typename Scalar, int Dim>
void MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::initParticleBucket(const std::set<unsigned int> &objects, ArrayND<std::map<unsigned int, std::vector<unsigned int> >,Dim> &bucket) const
{
    MPMSolid<Scalar,Dim> *mpm_solid_driver = dynamic_cast<MPMSolid<Scalar,Dim>*>(this->mpm_driver_);
    PHYSIKA_ASSERT(mpm_solid_driver);
    const Grid<Scalar,Dim> &grid = mpm_solid_driver->grid();
    Vector<unsigned int,Dim> grid_cell_num = grid.cellNum();
    //resize the bucket if necessary
    if(bucket.size() != grid_cell_num)
        bucket.resize(grid_cell_num);
    //clear the bucket entries
    for(typename ArrayND<std::map<unsigned int, std::vector<unsigned int> >,Dim>::Iterator iter = bucket.begin(); iter != bucket.end(); ++iter)
        (*iter).clear();
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
Vector<Scalar,Dim> MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::tangentialDirection(const Vector<Scalar,Dim> &normal, const Vector<Scalar,Dim> &velocity_diff) const
{
    Vector<Scalar,Dim> tangent_dir = velocity_diff - velocity_diff.dot(normal)*normal;
    tangent_dir.normalize();
    return tangent_dir;
}

template <typename Scalar, int Dim>
void MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::adjacentCells(const Vector<unsigned int,Dim> &node_idx,
                                                                     const Vector<unsigned int,Dim> &cell_num,
                                                                     std::vector<Vector<unsigned int,Dim> > &cells) const
{
    cells.clear();
    Vector<unsigned int,Dim> cell_idx = node_idx;
    switch(Dim)
    {
    case 2:
    {
        for(unsigned int offset_x = 0; offset_x <= 1; ++offset_x)
            for(unsigned int offset_y = 0; offset_y <= 1; ++offset_y)
            {
                cell_idx[0] = node_idx[0] - offset_x;
                cell_idx[1] = node_idx[1] - offset_y;
                if(cell_idx[0] < 0 || cell_idx[1] < 0 || cell_idx[0] >= cell_num[0] || cell_idx[1] >= cell_num[1])
                    continue;
                cells.push_back(cell_idx);
            }
        break;
    }
    case 3:
    {
        for(unsigned int offset_x = 0; offset_x <= 1; ++offset_x)
            for(unsigned int offset_y = 0; offset_y <= 1; ++offset_y)
                for(unsigned int offset_z = 0; offset_z <= 1; ++offset_z)
            {
                cell_idx[0] = node_idx[0] - offset_x;
                cell_idx[1] = node_idx[1] - offset_y;
                cell_idx[2] = node_idx[2] - offset_z;
                if(cell_idx[0] < 0 || cell_idx[1] < 0 || cell_idx[2] < 0
                    ||cell_idx[0] >= cell_num[0] || cell_idx[1] >= cell_num[1] || cell_idx[2] >= cell_num[2])
                    continue;
                cells.push_back(cell_idx);
            }
        break;
    }
    default:
        PHYSIKA_ERROR("Wrong dimension specified!");
    }
}


template <typename Scalar, int Dim>
void MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::resolveContactBetweenTwoObjects(const Vector<unsigned int,Dim> &node_idx, unsigned int object_idx1, unsigned int object_idx2,
                                                                                       const Vector<Scalar,Dim> &object1_normal_at_node, const Vector<Scalar,Dim> &object2_normal_at_node,
                                                                                       unsigned char is_object1_dirichlet_at_node, unsigned char is_object2_dirichlet_at_node, Scalar dt,
                                                                                       Vector<Scalar,Dim> &object1_node_velocity_delta, Vector<Scalar,Dim> &object2_node_velocity_delta)
{
    MPMSolid<Scalar,Dim> *mpm_solid_driver = dynamic_cast<MPMSolid<Scalar,Dim>*>(this->mpm_driver_);
    PHYSIKA_ASSERT(mpm_solid_driver);
    //clear data
    object1_node_velocity_delta = Vector<Scalar,Dim>(0);
    object2_node_velocity_delta = Vector<Scalar,Dim>(0);       
    Vector<Scalar,Dim> obj1_normal = object1_normal_at_node, obj2_normal = object2_normal_at_node; //normal will be modified to be colinear
    //resolve contact
    const Grid<Scalar,Dim> &grid = mpm_solid_driver->grid();
    Vector<unsigned int,Dim> grid_cell_num = grid.cellNum();
    Vector<Scalar,Dim> node_pos = grid.node(node_idx);
    Scalar obj1_mass = mpm_solid_driver->gridMass(object_idx1,node_idx), obj2_mass = mpm_solid_driver->gridMass(object_idx2,node_idx);
    Vector<Scalar,Dim> obj1_vel = mpm_solid_driver->gridVelocity(object_idx1,node_idx), obj2_vel = mpm_solid_driver->gridVelocity(object_idx2,node_idx);
    //average the normal of the two objects so that they're in opposite direction
    obj1_normal = (obj1_normal - obj2_normal).normalize();
    obj2_normal = -obj1_normal;
    Vector<Scalar,Dim> vel_delta = obj1_vel - obj2_vel;
    if((obj1_vel-obj2_vel).dot(obj1_normal) > 0) //necessary condition 1: approach each other
    {     
        //first compute the velocity after contact with no slip condition
        Vector<Scalar,Dim> obj1_new_vel(0), obj2_new_vel(0);
        if(is_object1_dirichlet_at_node) //if any of the two objects is dirichlet at the node, treat it's mass as infinity
        {
            obj1_new_vel = obj1_vel;
            obj2_new_vel = obj1_vel + restitution_coefficient_ * (obj1_vel - obj2_vel);
        }
        else if(is_object2_dirichlet_at_node)
        {
            obj1_new_vel = obj2_vel + restitution_coefficient_ * (obj2_vel - obj1_vel);
            obj2_new_vel = obj2_vel;
        }
        else
        {
            obj1_new_vel = (obj1_mass * obj1_vel + obj2_mass * obj2_vel + restitution_coefficient_ * obj2_mass * (obj2_vel - obj1_vel))/(obj1_mass + obj2_mass);
            obj2_new_vel = (obj1_mass * obj1_vel + obj2_mass * obj2_vel + restitution_coefficient_ * obj1_mass * (obj1_vel - obj2_vel))/(obj1_mass + obj2_mass);
        }
        //approximate the distance between objects with minimum distance between particles along the normal direction
        std::vector<Vector<unsigned int,Dim> > adjacent_cells;
        adjacentCells(node_idx,grid_cell_num,adjacent_cells);
        std::vector<unsigned int> particles_obj1;
        std::vector<unsigned int> particles_obj2;
        for(unsigned int i = 0; i < adjacent_cells.size(); ++i)
        {
            Vector<unsigned int,Dim> cell_idx = adjacent_cells[i];
            std::vector<unsigned int> &particles_in_cell_obj1 = particle_bucket_(cell_idx)[object_idx1];
            std::vector<unsigned int> &particles_in_cell_obj2 = particle_bucket_(cell_idx)[object_idx2];
            particles_obj1.insert(particles_obj1.end(),particles_in_cell_obj1.begin(),particles_in_cell_obj1.end());
            particles_obj2.insert(particles_obj2.end(),particles_in_cell_obj2.begin(),particles_in_cell_obj2.end());
        }
        Scalar min_dist = (std::numeric_limits<Scalar>::max)();
        for(unsigned int i = 0; i < particles_obj1.size(); ++i)
        {
            const SolidParticle<Scalar,Dim> &particle1 = mpm_solid_driver->particle(object_idx1,particles_obj1[i]);
            for(unsigned int j = 0; j < particles_obj2.size(); ++j)
            {
                const SolidParticle<Scalar,Dim> &particle2 = mpm_solid_driver->particle(object_idx2,particles_obj2[j]);
                //Scalar dist = (particle1.position() - particle2.position()).dot(obj1_normal);
                Scalar dist = (particle1.position() - particle2.position()).norm();
                if(dist < 0)
                    dist = -dist;
                if(dist < min_dist)
                    min_dist = dist;
            }
        }
        //resolve contact for each object at the node
        Scalar dist_threshold = collide_threshold_ * grid.minEdgeLength();
        if(min_dist < dist_threshold) //necessary condition 2: objects close enough
        {
            //compute the tangential direction
            Vector<Scalar,Dim> obj1_vel_delta = obj1_new_vel - obj1_vel, obj2_vel_delta = obj2_new_vel - obj2_vel;
            Vector<Scalar,Dim> obj1_tangent_dir = tangentialDirection(obj1_normal,obj1_vel_delta);
            Vector<Scalar,Dim> obj2_tangent_dir = -obj1_tangent_dir;
            //velocity difference in normal direction and tangential direction
            Scalar obj1_vel_delta_dot_norm = obj1_vel_delta.dot(obj1_normal);
            Scalar obj2_vel_delta_dot_norm = obj2_vel_delta.dot(obj2_normal);
            Scalar obj1_vel_delta_dot_tan = obj1_vel_delta.dot(obj1_tangent_dir);
            Scalar obj2_vel_delta_dot_tan = obj2_vel_delta.dot(obj2_tangent_dir);
            Vector<Scalar,Dim> obj1_vel_delta_norm = obj1_vel_delta_dot_norm * obj1_normal;
            Vector<Scalar,Dim> obj2_vel_delta_norm = obj2_vel_delta_dot_norm * obj2_normal;
            Vector<Scalar,Dim> obj1_vel_delta_tan = obj1_vel_delta_dot_tan * obj1_tangent_dir;
            Vector<Scalar,Dim> obj2_vel_delta_tan = obj2_vel_delta_dot_tan * obj2_tangent_dir;
            //slip with friction
            if(abs(obj1_vel_delta_dot_tan) > friction_coefficient_ * abs(obj1_vel_delta_dot_norm))
                obj1_vel_delta_tan = friction_coefficient_ * abs(obj1_vel_delta_dot_norm) * obj1_tangent_dir;
            if(abs(obj2_vel_delta_dot_tan) > friction_coefficient_ * abs(obj2_vel_delta_dot_norm))
                obj2_vel_delta_tan = friction_coefficient_ * abs(obj2_vel_delta_dot_norm) * obj2_tangent_dir;
            //apply a penalty function in the normal direction
            Scalar penalty_factor = 1 - pow(min_dist/dist_threshold,penalty_power_);
            obj1_vel_delta_norm *= penalty_factor;
            obj2_vel_delta_norm *= penalty_factor;
            //get the grid velocity impulse for each object
            if(is_object1_dirichlet_at_node == 0x00)
                object1_node_velocity_delta = obj1_vel_delta_norm + obj1_vel_delta_tan;
            if(is_object2_dirichlet_at_node == 0x00)
                object2_node_velocity_delta = obj2_vel_delta_norm + obj2_vel_delta_tan;
        }
    }
   
}

//explicit instantiations
template class MPMSolidSubgridFrictionContactMethod<float,2>;
template class MPMSolidSubgridFrictionContactMethod<float,3>;
template class MPMSolidSubgridFrictionContactMethod<double,2>;
template class MPMSolidSubgridFrictionContactMethod<double,3>;

}  //end of namespace Physika
