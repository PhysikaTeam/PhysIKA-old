/*
 * @file  collision_detection_result.cpp
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

#include "Physika_Dynamics/Collidable_Objects/collision_pair.h"
#include "Physika_Dynamics/Collidable_Objects/collision_detection_result.h"
#include "Physika_Dynamics/Collidable_Objects/mesh_based_collidable_object.h"

namespace Physika{

template <typename Scalar,int Dim>
CollisionDetectionResult<Scalar, Dim>::CollisionDetectionResult():
	number_pcs_(0)
{
}

template <typename Scalar,int Dim>
CollisionDetectionResult<Scalar, Dim>::~CollisionDetectionResult()
{
	cleanCollisionPairs();
}

template <typename Scalar,int Dim>
unsigned int CollisionDetectionResult<Scalar, Dim>::numberPCS() const
{
	return number_pcs_;
}

template <typename Scalar,int Dim>
unsigned int CollisionDetectionResult<Scalar, Dim>::numberCollision() const
{
	unsigned int number_collision = static_cast<unsigned int>(collision_pairs_.size());
	return number_collision;
}

template <typename Scalar,int Dim>
const std::vector<CollisionPairBase<Scalar, Dim>*>& CollisionDetectionResult<Scalar, Dim>::collisionPairs() const
{
	return collision_pairs_;
}

template <typename Scalar,int Dim>
std::vector<CollisionPairBase<Scalar, Dim>*>& CollisionDetectionResult<Scalar, Dim>::collisionPairs()
{
	return collision_pairs_;
}

template <typename Scalar,int Dim>
void CollisionDetectionResult<Scalar, Dim>::addPCS()
{
	number_pcs_++;
}

template <typename Scalar,int Dim>
void CollisionDetectionResult<Scalar, Dim>::cleanPCS()
{
	number_pcs_ = 0;
}

template <typename Scalar,int Dim>
void CollisionDetectionResult<Scalar, Dim>::addCollisionPair(CollisionPairBase<Scalar, Dim>* collision_pair)
{
	collision_pairs_.push_back(collision_pair);
}

template <typename Scalar,int Dim>
void CollisionDetectionResult<Scalar, Dim>::cleanCollisionPairs()
{
	unsigned int number_collision = static_cast<unsigned int>(collision_pairs_.size());
	for(unsigned int i = 0 ; i < number_collision; ++i)
	{
		if(collision_pairs_[i] != NULL)
		{
			delete collision_pairs_[i];
			collision_pairs_[i] = NULL;
		}
	}
	collision_pairs_.clear();
}

template <typename Scalar,int Dim>
void CollisionDetectionResult<Scalar, Dim>::addCollisionPair(MeshBasedCollidableObject<Scalar, Dim>* object_lhs, MeshBasedCollidableObject<Scalar, Dim>* object_rhs, unsigned int face_lhs_index, unsigned int face_rhs_index)
{
	CollisionPairMeshToMesh<Scalar, Dim>* collision_pair = new CollisionPairMeshToMesh<Scalar, Dim>(object_lhs, object_rhs, face_lhs_index, face_rhs_index);
	collision_pairs_.push_back(collision_pair);
}

template <typename Scalar,int Dim>
void CollisionDetectionResult<Scalar, Dim>::resetCollisionResults()
{
	cleanPCS();
	cleanCollisionPairs();
}

template class CollisionDetectionResult<float, 3>;
template class CollisionDetectionResult<double, 3>;

}