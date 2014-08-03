/*
 * @file  mesh_based_collidable_object.h
 * @collidable object based on the mesh of object
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

#ifndef PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_MESH_BASED_COLLIDABLE_OBJECT_H_
#define PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_MESH_BASED_COLLIDABLE_OBJECT_H_

#include "Physika_Dynamics/Collidable_Objects/collidable_object.h"

namespace Physika{

template <typename Scalar,int Dim> class Vector;
template <typename Scalar> class SurfaceMesh;
template <typename Scalar,int Dim> class CollisionPairManager;
template <typename Scalar,int Dim> class Transform;

template <typename Scalar>
class MeshBasedCollidableObject: public CollidableObject<Scalar, 3>
{
public:
	//constructors && deconstructors
	MeshBasedCollidableObject();
	~MeshBasedCollidableObject();

	//get & set
	typename CollidableObjectInternal::ObjectType objectType() const;
	const SurfaceMesh<Scalar>* mesh() const;
	SurfaceMesh<Scalar>* mesh();
	void setMesh(SurfaceMesh<Scalar>* mesh);
	const Transform<Scalar, 3>* transform() const;
	Transform<Scalar, 3>* transform();
	void setTransform(Transform<Scalar, 3>* transform);

    //return vertex and normals after transform
    Vector<Scalar, 3> vertexPosition(unsigned int vertex_index) const;
    Vector<Scalar, 3> faceNormal(unsigned int face_index) const;

	bool collideWithMesh(MeshBasedCollidableObject<Scalar>* object, unsigned int face_index_lhs, unsigned int face_index_rhs);

    //overlapPoint is the position of overlap. It will be changed after test if overlap is true.
	static bool overlapEdgeTriangle(const Vector<Scalar, 3>& vertex_edge_a, const Vector<Scalar, 3>& vertex_edge_b, const Vector<Scalar, 3>& vertex_face_a, const Vector<Scalar, 3>& vertex_face_b, const Vector<Scalar, 3>& vertex_face_c, Vector<Scalar, 3>& overlap_point);
	static bool overlapEdgeQuad(const Vector<Scalar, 3>& vertex_edge_a, const Vector<Scalar, 3>& vertex_edge_b, const Vector<Scalar, 3>& vertex_face_a, const Vector<Scalar, 3>& vertex_face_b, const Vector<Scalar, 3>& vertex_face_c, const Vector<Scalar, 3>& vertex_face_d, Vector<Scalar, 3>& overlap_point);
	
protected:
	//mesh_ is used to define the shape of object, while transform_ is used to define the configuration
	//For deformable object which only updates mesh_, transform_ can be set to identity
	SurfaceMesh<Scalar>* mesh_;
	Transform<Scalar, 3>* transform_;

};

}  //end of namespace Physika

#endif  //PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_MESH_BASED_COLLIDABLE_OBJECT_H_
