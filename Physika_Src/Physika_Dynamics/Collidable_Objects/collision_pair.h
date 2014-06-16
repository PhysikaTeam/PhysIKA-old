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

namespace Physika{

template <typename Scalar,int Dim> class CollisionPairBase;
template <typename Scalar,int Dim> class CollidableObject;
template <typename Scalar,int Dim> class MeshBasedCollidableObject;
template <typename Scalar> class Face;

template <typename Scalar,int Dim>
class CollisionPairBase
{
public:
	CollisionPairBase();
	virtual ~CollisionPairBase();
	virtual const CollidableObject<Scalar, Dim>* ObjectLhs() const = 0;
	virtual CollidableObject<Scalar, Dim>* ObjectLhs() = 0;
	virtual const CollidableObject<Scalar, Dim>* ObjectRhs() const = 0;
	virtual CollidableObject<Scalar, Dim>* ObjectRhs() = 0;
	virtual const MeshBasedCollidableObject<Scalar, Dim>* MeshObjectLhs() const = 0;
	virtual MeshBasedCollidableObject<Scalar, Dim>* MeshObjectLhs() = 0;
	virtual const MeshBasedCollidableObject<Scalar, Dim>* MeshObjectRhs() const = 0;
	virtual MeshBasedCollidableObject<Scalar, Dim>* MeshObjectRhs() = 0;
	virtual const Face<Scalar>* FaceLhs() const = 0;
	virtual Face<Scalar>* FaceLhs() = 0;
	virtual const Face<Scalar>* FaceRhs() const = 0;
	virtual Face<Scalar>* FaceRhs() = 0;
};

template <typename Scalar,int Dim>
class CollisionPairMesh2Mesh : public CollisionPairBase<Scalar, Dim>
{
public:
	CollisionPairMesh2Mesh(MeshBasedCollidableObject<Scalar, Dim>* object_lhs, MeshBasedCollidableObject<Scalar, Dim>* object_rhs, Face<Scalar>* face_lhs, Face<Scalar>* face_rhs);
	~CollisionPairMesh2Mesh();
	const CollidableObject<Scalar, Dim>* ObjectLhs() const;
	CollidableObject<Scalar, Dim>* ObjectLhs();
	const CollidableObject<Scalar, Dim>* ObjectRhs() const;
	CollidableObject<Scalar, Dim>* ObjectRhs();
	const MeshBasedCollidableObject<Scalar, Dim>* MeshObjectLhs() const;
	MeshBasedCollidableObject<Scalar, Dim>* MeshObjectLhs();
	const MeshBasedCollidableObject<Scalar, Dim>* MeshObjectRhs() const;
	MeshBasedCollidableObject<Scalar, Dim>* MeshObjectRhs();
	const Face<Scalar>* FaceLhs() const;
	Face<Scalar>* FaceLhs();
	const Face<Scalar>* FaceRhs() const;
	Face<Scalar>* FaceRhs();

protected:
	MeshBasedCollidableObject<Scalar, Dim>* object_lhs_;
	MeshBasedCollidableObject<Scalar, Dim>* object_rhs_;
	Face<Scalar>* face_lhs_;
	Face<Scalar>* face_rhs_;
};

}  //end of namespace Physikas

#endif  //PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_COLLISION_PAIR_H_