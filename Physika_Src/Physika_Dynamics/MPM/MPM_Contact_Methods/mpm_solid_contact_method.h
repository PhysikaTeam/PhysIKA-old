/*
 * @file mpm_solid_contact_method.h 
 * @Brief base class of all contact methods for mpm solid with uniform background grid.
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

#ifndef PHYSIKA_DYNAMICS_MPM_MPM_CONTACT_METHODS_MPM_SOLID_CONTACT_METHOD_H_
#define PHYSIKA_DYNAMICS_MPM_MPM_CONTACT_METHODS_MPM_SOLID_CONTACT_METHOD_H_

#include <vector>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Dynamics/MPM/MPM_Contact_Methods/mpm_contact_method.h"

namespace Physika{

template <typename Scalar, int Dim>
class MPMSolidContactMethod: public MPMContactMethod<Scalar,Dim>
{
public:
    MPMSolidContactMethod();
    ~MPMSolidContactMethod();
    virtual MPMSolidContactMethod<Scalar,Dim>* clone() const = 0;
    //potential collide nodes are those where multiple velocity fields exist
    virtual void resolveContact(const std::vector<Vector<unsigned int,Dim> > &potential_collide_nodes,
                                const std::vector<std::vector<unsigned int> > &objects_at_node,
                                const std::vector<std::vector<Vector<Scalar,Dim> > > &normal_at_node,
                                Scalar dt) = 0;
protected:
};


}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_MPM_CONTACT_METHODS_MPM_SOLID_CONTACT_METHOD_H_
