/*
 * @file  mesh_based_collidable_object.cpp
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

#include "Physika_Dynamics/Collidable_Objects/mesh_based_collidable_object.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Dynamics/Collidable_Objects/collision_detection_result.h"

namespace Physika{

template <typename Scalar,int Dim>
MeshBasedCollidableObject<Scalar, Dim>::MeshBasedCollidableObject():
	transform_(Vector<Scalar,3>(0, 0, 0))
{
}

template <typename Scalar,int Dim>
MeshBasedCollidableObject<Scalar, Dim>::~MeshBasedCollidableObject()
{
}

template <typename Scalar,int Dim>
typename CollidableObject<Scalar, Dim>::ObjectType MeshBasedCollidableObject<Scalar, Dim>::getObjectType() const
{
	return CollidableObject<Scalar, Dim>::MESH_BASED;
}

template <typename Scalar,int Dim>
const SurfaceMesh<Scalar>* MeshBasedCollidableObject<Scalar, Dim>::getMesh() const
{
	return mesh_;
}

template <typename Scalar,int Dim>
SurfaceMesh<Scalar>* MeshBasedCollidableObject<Scalar, Dim>::getMesh()
{
	return mesh_;
}


template <typename Scalar,int Dim>
void MeshBasedCollidableObject<Scalar, Dim>::setMesh(SurfaceMesh<Scalar>* mesh)
{
	mesh_ = mesh;
}

template <typename Scalar,int Dim>
Vector<Scalar, 3> MeshBasedCollidableObject<Scalar, Dim>::getPointPosition(unsigned int point_index) const
{
	return transform_.transform(mesh_->vertexPosition(point_index));
}

template <typename Scalar,int Dim>
const Transform<Scalar>& MeshBasedCollidableObject<Scalar, Dim>::transform() const
{
	return transform_;
}

template <typename Scalar,int Dim>
Transform<Scalar>& MeshBasedCollidableObject<Scalar, Dim>::transform()
{
	return transform_;
}

template <typename Scalar,int Dim>
void MeshBasedCollidableObject<Scalar, Dim>::setTransform(const Transform<Scalar>& transform)
{
	transform_ = transform;
}

template <typename Scalar,int Dim>
bool MeshBasedCollidableObject<Scalar, Dim>::collideWithMesh(MeshBasedCollidableObject<Scalar, Dim>* object, unsigned int face_index_lhs, unsigned int face_index_rhs, CollisionDetectionResult<Scalar, Dim>& collision_result)
{
	if(object == NULL || object->getMesh() == NULL)
		return false;

	Face<Scalar>& face_lhs = mesh_->face(face_index_lhs);
	Face<Scalar>& face_rhs = object->getMesh()->face(face_index_rhs);
	unsigned int num_vertex_lhs = face_lhs.numVertices();
	unsigned int num_vertex_rhs = face_rhs.numVertices();
	Vector<Scalar, 3>* vertex_lhs = new Vector<Scalar, 3>[num_vertex_lhs];
	Vector<Scalar, 3>* vertex_rhs = new Vector<Scalar, 3>[num_vertex_rhs];

	for(unsigned int i = 0; i < num_vertex_lhs; i++)
	{
		vertex_lhs[i] = getPointPosition(face_lhs.vertex(i).positionIndex());
	}
	for(unsigned int i = 0; i < num_vertex_rhs; i++)
	{
		vertex_rhs[i] = object->getPointPosition(face_rhs.vertex(i).positionIndex());
	}

	bool is_overlap = false;
	bool is_lhs_tri = (num_vertex_lhs == 3);
	bool is_rhs_tri = (num_vertex_rhs == 3);
	bool is_lhs_quad = (num_vertex_lhs == 4);
	bool is_rhs_quad = (num_vertex_rhs == 4);

	//test each edge of lhs with the face of rhs
	for(unsigned int i = 0; i < num_vertex_lhs; i++)
	{
		if(is_rhs_tri)//triangle
		{
			if(overlapEdgeTriangle(vertex_lhs[i], vertex_lhs[(i + 1)%num_vertex_lhs], vertex_rhs[0], vertex_rhs[1], vertex_rhs[2]))
				is_overlap = true;
		}
		if(is_rhs_quad)//quad
		{
			if(overlapEdgeQuad(vertex_lhs[i], vertex_lhs[(i + 1)%num_vertex_lhs], vertex_rhs[0], vertex_rhs[1], vertex_rhs[2], vertex_rhs[3]))
				is_overlap = true;
		}
	}
	//test each edge of rhs with the face of lhs
	for(unsigned int i = 0; i < num_vertex_rhs; i++)
	{
		if(is_lhs_tri)//triangle
		{
			if(overlapEdgeTriangle(vertex_rhs[i], vertex_rhs[(i + 1)%num_vertex_rhs], vertex_lhs[0], vertex_lhs[1], vertex_lhs[2]))
				is_overlap = true;
		}
		if(is_lhs_quad)//quad
		{
			if(overlapEdgeQuad(vertex_rhs[i], vertex_rhs[(i + 1)%num_vertex_rhs], vertex_lhs[0], vertex_lhs[1], vertex_lhs[2], vertex_lhs[3]))
				is_overlap = true;
		}
	}

	collision_result.addPCS();
	if(is_overlap)
	{
		collision_result.addCollisionPair(this, object, &face_lhs, &face_rhs);
	}

	delete[] vertex_lhs;
	delete[] vertex_rhs;
	return is_overlap;
}

template <typename Scalar,int Dim>
bool MeshBasedCollidableObject<Scalar, Dim>::overlapEdgeTriangle(const Vector<Scalar, 3>& vertex_edge_a, const Vector<Scalar, 3>& vertex_edge_b, const Vector<Scalar, 3>& vertex_face_a, const Vector<Scalar, 3>& vertex_face_b, const Vector<Scalar, 3>& vertex_face_c) const
{
	Vector<Scalar, 3> face_normal = (vertex_face_b - vertex_face_a).cross(vertex_face_c - vertex_face_a);
	face_normal.normalize();
	Scalar length = face_normal.dot(vertex_edge_b - vertex_edge_a);
	if(abs(length) < FLOAT_EPSILON)
		return false;
	Scalar ratio = face_normal.dot(vertex_face_a - vertex_edge_a)/length;
	if(ratio < 0 || ratio > 1)
		return false;
	Vector<Scalar, 3> projection = vertex_edge_a + (vertex_edge_b - vertex_edge_a) * ratio;
	bool inTriTestBA = (((vertex_face_b - vertex_face_a).cross(projection - vertex_face_a)).dot(face_normal) >= 0);
	bool inTriTestCB = (((vertex_face_c - vertex_face_b).cross(projection - vertex_face_b)).dot(face_normal) >= 0);
	bool inTriTestAC = (((vertex_face_a - vertex_face_c).cross(projection - vertex_face_c)).dot(face_normal) >= 0);
	if(inTriTestBA && inTriTestCB && inTriTestAC)
		return true;
	return false;
}

template <typename Scalar,int Dim>
bool MeshBasedCollidableObject<Scalar, Dim>::overlapEdgeQuad(const Vector<Scalar, 3>& vertex_edge_a, const Vector<Scalar, 3>& vertex_edge_b, const Vector<Scalar, 3>& vertex_face_a, const Vector<Scalar, 3>& vertex_face_b, const Vector<Scalar, 3>& vertex_face_c, const Vector<Scalar, 3>& vertex_face_d) const
{
	return false;
}

//explicit instantitation
template class MeshBasedCollidableObject<float, 3>;
template class MeshBasedCollidableObject<double, 3>;

}
