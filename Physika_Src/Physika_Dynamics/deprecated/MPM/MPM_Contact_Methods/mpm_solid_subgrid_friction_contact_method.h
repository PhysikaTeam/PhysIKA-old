/*
 * @file mpm_solid_subgrid_friction_contact_method.h 
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

#ifndef PHYSIKA_DYNAMICS_MPM_MPM_CONTACT_METHODS_MPM_SOLID_SUBGRID_FRICTION_CONTACT_METHOD_H_
#define PHYSIKA_DYNAMICS_MPM_MPM_CONTACT_METHODS_MPM_SOLID_SUBGRID_FRICTION_CONTACT_METHOD_H_

#include <set>
#include <map>
#include <vector>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Arrays/array_Nd.h"
#include "Physika_Dynamics/MPM/MPM_Contact_Methods/mpm_solid_contact_method.h"

namespace Physika{

/*
 * MPMSolidSubgridFrictionContactMethod:
 * 1. the Coulomb friction between objects is modeled
 * 2. the contact threshold is set with respect to the grid cell size such that resolution of 
 *    contact detection is increased to subgrid level
 * 3. when objects approach each other within the threshold, a penalty function is applied to 
 *    the velocity change in normal direction
 * 4. multiple objects can collide at a node with momentum conserved
 * 5. the contact varies from elastic to inealstic, controlled by the coefficient of restitution
 *
 */

template <typename Scalar, int Dim>
class MPMSolidSubgridFrictionContactMethod: public MPMSolidContactMethod<Scalar,Dim>
{
public:
    MPMSolidSubgridFrictionContactMethod();
    MPMSolidSubgridFrictionContactMethod(const MPMSolidSubgridFrictionContactMethod<Scalar,Dim> &contact_method);
    ~MPMSolidSubgridFrictionContactMethod();
    MPMSolidSubgridFrictionContactMethod<Scalar,Dim>& operator= (const MPMSolidSubgridFrictionContactMethod<Scalar,Dim> &contact_method);
    virtual MPMSolidSubgridFrictionContactMethod<Scalar,Dim>* clone() const;
    virtual void resolveContact(const std::vector<Vector<unsigned int,Dim> > &potential_collide_nodes,
                                const std::vector<std::vector<unsigned int> > &objects_at_node,
                                const std::vector<std::vector<Vector<Scalar,Dim> > > &normal_at_node,
                                const std::vector<std::vector<unsigned char> > &is_dirichlet_at_node,
                                Scalar dt);
    void setFrictionCoefficient(Scalar coefficient);
    void setCollideThreshold(Scalar threshold);
    void setPenaltyPower(Scalar penalty_power);
    void setRestitutionCoefficient(Scalar restitution_coefficient);
    Scalar frictionCoefficient() const;
    Scalar collideThreshold() const;
    Scalar penaltyPower() const;
    Scalar restitutionCoefficient() const;
protected:
    //put the particles of given object into buckets according to their positions, each bucket is a grid cell
    //the key of the map is the object id, and the value of the map is the vector of particles in this bucket
    void initParticleBucket(const std::set<unsigned int> &objects, ArrayND<std::map<unsigned int,std::vector<unsigned int> >,Dim> &bucket) const;
    //helper method to compute tangential direction from normal direction and the velocity difference  
    Vector<Scalar,Dim> tangentialDirection(const Vector<Scalar,Dim> &normal, const Vector<Scalar,Dim> &velocity_diff) const;
    //return indices of cells that is adjacent to given node
    void adjacentCells(const Vector<unsigned int,Dim> &node_idx, const Vector<unsigned int,Dim> &cell_num, std::vector<Vector<unsigned int,Dim> > &cells) const;
    //resolve contact between two objects, return the velocity impulse of applied on each object at the node
    void resolveContactBetweenTwoObjects(const Vector<unsigned int,Dim> &node_idx, unsigned int object_idx1, unsigned int object_idx2,
                                         const Vector<Scalar,Dim> &object1_normal_at_node, const Vector<Scalar,Dim> &object2_normal_at_node,
                                         unsigned char is_object1_dirichlet_at_node, unsigned char is_object2_dirichlet_at_node, Scalar dt,
                                         Vector<Scalar,Dim> &object1_node_velocity_delta, Vector<Scalar,Dim> &object2_node_velocity_delta);  
protected:
    Scalar friction_coefficient_;  //the coefficient between normal contact force and tangential frictional force 
    Scalar collide_threshold_;  //the collide distance threshold expressed with respect to grid element size, in range (0,1]
    Scalar penalty_power_; //controls the power of the penalty function
    Scalar restitution_coefficient_; //ranges from 0~1, controls the contact type [inelastic, elastic]
    ArrayND<std::map<unsigned int,std::vector<unsigned int> >,Dim> particle_bucket_;
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_MPM_CONTACT_METHODS_MPM_SOLID_SUBGRID_FRICTION_CONTACT_METHOD_H_
