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
#include <iostream>
#include "Physika_Core/Utilities/math_utilities.h"
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
    for(unsigned int i = 0; i <potential_collide_nodes.size(); ++i)
    {
        Vector<unsigned int,Dim> node_idx = potential_collide_nodes[i];
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
                //compute the tangential direction
                Vector<Scalar,Dim> tangent_dir = vel_delta.cross(normal_at_node[i][j]);
                tangent_dir.normalize();
                tangent_dir = normal_at_node[i][j].cross(tangent_dir);
                //velocity difference in normal direction and tangential direction
                Scalar vel_delta_dot_tan = vel_delta.dot(tangent_dir);
                Vector<Scalar,Dim> vel_delta_norm = vel_delta_dot_norm * normal_at_node[i][j];
                Vector<Scalar,Dim> vel_delta_tan = vel_delta_dot_tan * tangent_dir;
                if(abs(vel_delta_dot_tan) > friction_coefficient_ * abs(vel_delta_dot_norm)) //slip with friction
                    vel_delta_tan = friction_coefficient_ * vel_delta_norm;
                //apply a penaly function in the normal direction
                Scalar grid_edge_len = (mpm_solid_driver->grid()).minEdgeLength();
                Scalar dist_threshold = collide_threshold_ * grid_edge_len;
                //TO DO
                //update the grid velocity
                Vector<Scalar,Dim> new_vel = trial_vel - vel_delta_norm - vel_delta_tan;
                mpm_solid_driver->setGridVelocity(obj_idx,node_idx,new_vel);
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
    if(threshold < 0)
    {
        std::cout<<"Warning: invalid collide threshold, 0.5 of the grid cell edge length is used instead!\n";
        collide_threshold_ = 0.5;
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
void MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::initParticleBucket(const std::set<unsigned int> &objects, ArrayND<std::map<unsigned int, std::vector<unsigned int> >,Dim> &bucket)
{
    //TO DO
}

//explicit instantiations
//template class MPMSolidSubgridFrictionContactMethod<float,2>;
template class MPMSolidSubgridFrictionContactMethod<float,3>;
//template class MPMSolidSubgridFrictionContactMethod<double,2>;
template class MPMSolidSubgridFrictionContactMethod<double,3>;

}  //end of namespace Physika
