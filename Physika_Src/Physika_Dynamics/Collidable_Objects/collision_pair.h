/*
 * @file  collision_pair.h
 * @pairs of colliding elementaries
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

#ifndef PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_COLLISION_PAIR_H_
#define PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_COLLISION_PAIR_H_

#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"

namespace Physika{

template <typename Scalar,int Dim> class CollisionPairBase;
template <typename Scalar,int Dim> class CollidableObject;
template <typename Scalar,int Dim> class MeshBasedCollidableObject;

template <typename Scalar,int Dim>
class CollisionPairBase
{
public:
	CollisionPairBase();
	virtual ~CollisionPairBase();
	//Functions for getting objects and faces of a mesh-to-mesh collision pair, corresponding to class CollisionPairMesh2Mesh
	virtual const CollidableObject<Scalar, Dim>* objectLhs() const = 0;
	virtual CollidableObject<Scalar, Dim>* objectLhs() = 0;
	virtual const CollidableObject<Scalar, Dim>* objectRhs() const = 0;
	virtual CollidableObject<Scalar, Dim>* objectRhs() = 0;
	virtual const MeshBasedCollidableObject<Scalar, Dim>* meshObjectLhs() const = 0;
	virtual MeshBasedCollidableObject<Scalar, Dim>* meshObjectLhs() = 0;
	virtual const MeshBasedCollidableObject<Scalar, Dim>* meshObjectRhs() const = 0;
	virtual MeshBasedCollidableObject<Scalar, Dim>* meshObjectRhs() = 0;
	virtual const Face<Scalar>* faceLhs() const = 0;
	virtual Face<Scalar>* faceLhs() = 0;
	virtual const Face<Scalar>* faceRhs() const = 0;
	virtual Face<Scalar>* faceRhs() = 0;

	//If other kinds of collision pairs need to be defined, e.g. mesh-to-implicit pairs, add corresponding functions here
};

//Face pair of a mesh-to-mesh collision
template <typename Scalar,int Dim>
class CollisionPairMesh2Mesh : public CollisionPairBase<Scalar, Dim>
{
public:
	CollisionPairMesh2Mesh(MeshBasedCollidableObject<Scalar, Dim>* object_lhs, MeshBasedCollidableObject<Scalar, Dim>* object_rhs, Face<Scalar>* face_lhs, Face<Scalar>* face_rhs);
	~CollisionPairMesh2Mesh();
	const CollidableObject<Scalar, Dim>* objectLhs() const;
	CollidableObject<Scalar, Dim>* objectLhs();
	const CollidableObject<Scalar, Dim>* objectRhs() const;
	CollidableObject<Scalar, Dim>* objectRhs();
	const MeshBasedCollidableObject<Scalar, Dim>* meshObjectLhs() const;
	MeshBasedCollidableObject<Scalar, Dim>* meshObjectLhs();
	const MeshBasedCollidableObject<Scalar, Dim>* meshObjectRhs() const;
	MeshBasedCollidableObject<Scalar, Dim>* meshObjectRhs();
	const Face<Scalar>* faceLhs() const;
	Face<Scalar>* faceLhs();
	const Face<Scalar>* faceRhs() const;
	Face<Scalar>* faceRhs();

protected:
	MeshBasedCollidableObject<Scalar, Dim>* object_lhs_;
	MeshBasedCollidableObject<Scalar, Dim>* object_rhs_;
	Face<Scalar>* face_lhs_;
	Face<Scalar>* face_rhs_;
};

}  //end of namespace Physikas

#endif  //PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_COLLISION_PAIR_H_