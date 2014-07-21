/*
 * @file  contact_point_manager.h
 * @manager of contact points in rigid body simulation
 * @author Tianxiang Zhang
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_CONTACT_POINT_MANAGER_H_
#define PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_CONTACT_POINT_MANAGER_H_

#include <vector>

namespace Physika{

template <typename Scalar,int Dim> class CollisionPairManager;
template <typename Scalar> class CollisionPairMeshToMesh;
template <typename Scalar,int Dim> class CollisionPairBase;
template <typename Scalar,int Dim> class ContactPoint;

template <typename Scalar,int Dim>
class ContactPointManager
{
public:
    ContactPointManager();
    ~ContactPointManager();

    //get & set
    void setCollisionResult(CollisionPairManager<Scalar, Dim>& collision_result);
    unsigned int numContactPoint() const;
    ContactPoint<Scalar, Dim>* contactPoint(unsigned int contact_index);
    const std::vector<ContactPoint<Scalar, Dim>* >& contactPoints() const;
    std::vector<ContactPoint<Scalar, Dim>* >& contactPoints();
    ContactPoint<Scalar, Dim>* operator[] (unsigned int contact_index);

    //clean contact points
    void cleanContactPoints();

protected:
    std::vector<ContactPoint<Scalar, Dim>* > contact_points_;

    //contact sampling
    void getMeshContactPoint(CollisionPairMeshToMesh<Scalar>* collision_pair);
};

} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_CONTACT_POINT_MANAGER_H_