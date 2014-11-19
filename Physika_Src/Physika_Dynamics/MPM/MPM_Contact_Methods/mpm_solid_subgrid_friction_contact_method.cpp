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

#include <iostream>
#include "Physika_Dynamics/MPM/MPM_Contact_Methods/mpm_solid_subgrid_friction_contact_method.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::MPMSolidSubgridFrictionContactMethod()
    :MPMSolidContactMethod<Scalar,Dim>(),friction_coefficient_(0.5),collide_threshold_(0.5)
{
}

template <typename Scalar, int Dim>
MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::~MPMSolidSubgridFrictionContactMethod()
{
}

template <typename Scalar, int Dim>
void MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::resolveContact(const std::vector<Vector<unsigned int,Dim> > &potential_collide_nodes, Scalar dt)
{
//TO DO
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
Scalar MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::frictionCoefficient() const
{
    return friction_coefficient_;
}

template <typename Scalar, int Dim>
Scalar MPMSolidSubgridFrictionContactMethod<Scalar,Dim>::collideThreshold() const
{
    return collide_threshold_;
}

//explicit instantiations
template class MPMSolidSubgridFrictionContactMethod<float,2>;
template class MPMSolidSubgridFrictionContactMethod<float,3>;
template class MPMSolidSubgridFrictionContactMethod<double,2>;
template class MPMSolidSubgridFrictionContactMethod<double,3>;

}  //end of namespace Physika
