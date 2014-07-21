/*
 * @file  collision_pair_manager.h
 * @results of collision detection
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

#ifndef PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_COLLISION_PAIR_MANAGER_H_
#define PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_COLLISION_PAIR_MANAGER_H_

#include <vector>
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"

namespace Physika{

template <typename Scalar,int Dim> class CollisionPairBase;
template <typename Scalar> class MeshBasedCollidableObject;

template <typename Scalar,int Dim>
class CollisionPairManager
{
public:
	//constructors && deconstructors
	CollisionPairManager();
	~CollisionPairManager();

	//get
	unsigned int numberPCS() const;
	unsigned int numberCollision() const;
	const std::vector<CollisionPairBase<Scalar, Dim>*>& collisionPairs() const;
	std::vector<CollisionPairBase<Scalar, Dim>*>& collisionPairs();

	//structure maintain
	void setCurrentObjectIndex(unsigned int current_object_lhs_idx, unsigned int current_object_rhs_idx);
	void addPCS();
	void cleanPCS();
	void addCollisionPair(CollisionPairBase<Scalar, Dim>* collision_pair);
	void cleanCollisionPairs();

	void addCollisionPair(MeshBasedCollidableObject<Scalar>* object_lhs, MeshBasedCollidableObject<Scalar>* object_rhs, unsigned int face_lhs_index, unsigned int face_rhs_index);

	//clean PCS and collision pairs
	void resetCollisionResults();

protected:
	//Potential Collide Set (PCS) contains pairs whose bounding volumes overlap.
	//Generally PCS doesn't need to be recorded in detail, therefor a simple variable is defined here to count the number of it.
	//PCS is only used in the statistics of a collision detection algorithm. For normal use of collision detection, PCS can be ignored.
	unsigned int number_pcs_;

	//Index of current objects. They can be set by function setCurrentObjectIndex before collision detection.
	//When adding collision pairs, they will be added into the pairs.
	unsigned int current_object_lhs_idx_, current_object_rhs_idx_;
	std::vector<CollisionPairBase<Scalar, Dim>*> collision_pairs_;
};

}  //end of namespace Physikas

#endif  //PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_COLLISION_PAIR_MANAGER_H_