/*
 * @file  object_bvh_node.h
 * @node of a collidable object's BVH
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

#ifndef PHYSIKA_GEOMETRY_BOUNDING_VOLUME_OBJECT_BVH_NODE_H_
#define PHYSIKA_GEOMETRY_BOUNDING_VOLUME_OBJECT_BVH_NODE_H_

#include "Physika_Geometry/Boundary_Meshes/surface_mesh.h"
#include "Physika_Dynamics/Collidable_Objects/collidable_object.h"

namespace Physika{

template <typename Scalar,int Dim> class Vector;
template <typename Scalar,int Dim> class BVHNodeBase;
template <typename Scalar,int Dim> class CollisionPairManager;

template <typename Scalar,int Dim>
class ObjectBVHNode : public BVHNodeBase<Scalar, Dim>
{
public:
	//constructors && deconstructors
	ObjectBVHNode();
	~ObjectBVHNode();

	//get & set
	bool isSceneNode() const;
	bool isObjectNode() const;
	typename CollidableObjectInternal::ObjectType objectType() const;
	void setObject(CollidableObject<Scalar, Dim>* object);
	const CollidableObject<Scalar, Dim>* object() const;
	void setFaceIndex(unsigned int face_index);
	unsigned int getFaceIndex() const;

	//structure maintain
	void resize();

	bool elemTest(const BVHNodeBase<Scalar, Dim>* const target, CollisionPairManager<Scalar, Dim>& collision_result);
	
protected:
	typename CollidableObjectInternal::ObjectType object_type_;
	CollidableObject<Scalar, Dim>* object_;
	unsigned int face_index_;
	bool has_face_;

    void buildFromFace();
};

}  //end of namespace Physika

#endif  //PHYSIKA_GEOMETRY_BOUNDING_VOLUME_OBJECT_BVH_NODE_H_
