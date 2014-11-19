/*
 * @file mpm_solid_subgrid_friction_contact_method.h 
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

#ifndef PHYSIKA_DYNAMICS_MPM_MPM_CONTACT_METHODS_MPM_SOLID_SUBGRID_FRICTION_CONTACT_METHOD_H_
#define PHYSIKA_DYNAMICS_MPM_MPM_CONTACT_METHODS_MPM_SOLID_SUBGRID_FRICTION_CONTACT_METHOD_H_

#include <vector>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Dynamics/MPM/MPM_Contact_Methods/mpm_solid_contact_method.h"

namespace Physika{

/*
 * MPMSolidSubgridFrictionContactMethod:
 * 1. the Coulomb friction between objects is modeled
 * 2. the contact threshold is set with respect to the grid cell size such that resolution of 
 *    contact detection is increased to subgrid level
 */

template <typename Scalar, int Dim>
class MPMSolidSubgridFrictionContactMethod: public MPMSolidContactMethod<Scalar,Dim>
{
public:
    MPMSolidSubgridFrictionContactMethod();
    ~MPMSolidSubgridFrictionContactMethod();
    virtual void resolveContact(const std::vector<Vector<unsigned int,Dim> > &potential_collide_nodes, Scalar dt);
    void setFrictionCoefficient(Scalar coefficient);
    void setCollideThreshold(Scalar threshold);
    Scalar frictionCoefficient() const;
    Scalar collideThreshold() const;
protected:
    Scalar friction_coefficient_;  //the coefficient between normal contact force and tangential frictional force 
    Scalar collide_threshold_;  //the collide distance threshold expressed with respect to grid element size
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_MPM_CONTACT_METHODS_MPM_SOLID_SUBGRID_FRICTION_CONTACT_METHOD_H_
