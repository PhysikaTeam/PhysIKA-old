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

#include "Physika_Geometry/Boundary_Meshes/surface_mesh.h"
#include "Physika_Dynamics/Collidable_Objects/collidable_object.h"

namespace Physika{

template <typename Scalar,int Dim> class CollisionPairBase;
template <typename Scalar,int Dim> class CollidableObject;
template <typename Scalar> class MeshBasedCollidableObject;

/*
 * CollisionPair contains colliding elements (e.g. faces of meshes) and objects.
 * It's different from ContactPoint, which contains position and normal of contact points.
 * ContactPoint should be generated from CollisionPair. This process is called "contact sampling".
 * Notice that contact sampling is not a bijection.
 */

template <typename Scalar,int Dim>
class CollisionPairBase
{
public:
	CollisionPairBase();
	virtual ~CollisionPairBase();

    virtual typename CollidableObjectInternal::ObjectType objectTypeLhs() const = 0;
    virtual typename CollidableObjectInternal::ObjectType objectTypeRhs() const = 0;

    virtual const CollidableObject<Scalar, Dim>* objectLhs() const = 0;
	virtual CollidableObject<Scalar, Dim>* objectLhs() = 0;
	virtual const CollidableObject<Scalar, Dim>* objectRhs() const = 0;
	virtual CollidableObject<Scalar, Dim>* objectRhs() = 0;
	virtual unsigned int objectLhsIdx() const = 0;
	virtual unsigned int objectRhsIdx() const = 0;

};

//Face pair of a mesh-to-mesh collision
template <typename Scalar>
class CollisionPairMeshToMesh : public CollisionPairBase<Scalar, 3>
{
public:
	CollisionPairMeshToMesh(unsigned int object_lhs_index, unsigned int object_rhs_index,
							MeshBasedCollidableObject<Scalar>* object_lhs, MeshBasedCollidableObject<Scalar>* object_rhs,
							unsigned int face_lhs_index, unsigned int face_rhs_index);
	~CollisionPairMeshToMesh();

    typename CollidableObjectInternal::ObjectType objectTypeLhs() const;
    typename CollidableObjectInternal::ObjectType objectTypeRhs() const;

	const CollidableObject<Scalar, 3>* objectLhs() const;
	CollidableObject<Scalar, 3>* objectLhs();
	const CollidableObject<Scalar, 3>* objectRhs() const;
	CollidableObject<Scalar, 3>* objectRhs();

	const MeshBasedCollidableObject<Scalar>* meshObjectLhs() const;
	MeshBasedCollidableObject<Scalar>* meshObjectLhs();
	const MeshBasedCollidableObject<Scalar>* meshObjectRhs() const;
	MeshBasedCollidableObject<Scalar>* meshObjectRhs();
	const SurfaceMeshInternal::Face<Scalar>* faceLhsPtr() const;
	SurfaceMeshInternal::Face<Scalar>* faceLhsPtr();
	const SurfaceMeshInternal::Face<Scalar>* faceRhsPtr() const;
	SurfaceMeshInternal::Face<Scalar>* faceRhsPtr();
	unsigned int faceLhsIdx() const;
	unsigned int faceRhsIdx() const;

	unsigned int objectLhsIdx() const;
	unsigned int objectRhsIdx() const;

protected:
	unsigned int object_lhs_index_;
	unsigned int object_rhs_index_;
	MeshBasedCollidableObject<Scalar>* object_lhs_;
	MeshBasedCollidableObject<Scalar>* object_rhs_;
	unsigned int face_lhs_index_;
	unsigned int face_rhs_index_;
	SurfaceMeshInternal::Face<Scalar>* face_lhs_;
	SurfaceMeshInternal::Face<Scalar>* face_rhs_;
};

}  //end of namespace Physika

#endif  //PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_COLLISION_PAIR_H_
